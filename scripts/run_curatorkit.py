#!/usr/bin/env python3
"""CuratorKIT validation, deduplication, and leak checking for TurnDrift-Control.

Runs schema validation, exact dedup, and MinHash near-duplicate detection
on the benchmark dataset. Reports before/after counts and cross-split
near-duplicate checks (dev vs test, calibration vs test).

Usage:
    python scripts/run_curatorkit.py [--out results/curatorkit]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig, DataSample
from curatorkit.gates import DiversityGate, SchemaGate


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def episode_to_text(ep: dict) -> str:
    parts = [ep.get("system_message", "")]
    for turn in ep.get("turns", []):
        parts.append(f"{turn['speaker']}: {turn['content']}")
    return "\n".join(parts)


def episode_to_sample(ep: dict) -> DataSample:
    return DataSample(
        source_uri=f"turndrift_core::{ep['episode_id']}",
        instruction=ep.get("system_message", ""),
        output=episode_to_text(ep),
        metadata={
            "episode_id": ep["episode_id"],
            "variant": ep["variant"],
            "split": ep["split"],
            "domain": ep["scenario_domain"],
            "strategy": ep["injection_strategy"],
            "target": ep["injection_target"],
        },
    )


def validate_schema(episodes: list[dict], schema_path: Path) -> dict:
    import jsonschema

    schema = json.load(open(schema_path))
    errors = []
    for ep in episodes:
        try:
            jsonschema.validate(ep, schema)
        except jsonschema.ValidationError as e:
            errors.append({"episode_id": ep.get("episode_id", "?"), "error": str(e.message)[:200]})
    return {
        "total": len(episodes),
        "valid": len(episodes) - len(errors),
        "invalid": len(errors),
        "errors": errors,
    }


def run_schema_gate(samples: list[DataSample]) -> dict:
    gate = SchemaGate(
        required_fields=["instruction", "output"],
        min_tokens=20,
        max_tokens=8192,
    )
    passed, rejected = gate.run(samples)
    return {
        "before": len(samples),
        "after": len(passed),
        "rejected": len(rejected),
        "rejection_reasons": [
            {"episode_id": r.metadata.get("episode_id", "?"), "reason": r.rejection_reason}
            for r in rejected[:10]
        ],
    }


def run_exact_dedup(episodes: list[dict]) -> dict:
    seen = {}
    duplicates = []
    for ep in episodes:
        text = episode_to_text(ep)
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            duplicates.append((ep["episode_id"], seen[h]))
        else:
            seen[h] = ep["episode_id"]
    return {
        "total": len(episodes),
        "unique": len(seen),
        "duplicates": len(duplicates),
        "duplicate_pairs": duplicates,
    }


def run_minhash_dedup(samples: list[DataSample], threshold: float = 0.85) -> dict:
    gate = DiversityGate(
        similarity_threshold=threshold,
        text_field="output",
    )
    passed, rejected = gate.run(samples)
    return {
        "before": len(samples),
        "after": len(passed),
        "near_duplicates_removed": len(rejected),
        "threshold": threshold,
        "removed": [
            {"episode_id": r.metadata.get("episode_id", "?"), "reason": r.rejection_reason}
            for r in rejected[:20]
        ],
    }


def cross_split_leak_check(
    split_a: list[DataSample],
    split_b: list[DataSample],
    name_a: str,
    name_b: str,
    threshold: float = 0.85,
) -> dict:
    gate = DiversityGate(
        similarity_threshold=threshold,
        text_field="output",
    )
    combined = split_a + split_b
    passed, rejected = gate.run(combined)
    cross_leaks = []
    a_ids = {s.metadata["episode_id"] for s in split_a}
    b_ids = {s.metadata["episode_id"] for s in split_b}
    for r in rejected:
        eid = r.metadata.get("episode_id", "?")
        if eid in b_ids:
            cross_leaks.append({"episode_id": eid, "reason": r.rejection_reason})

    return {
        "split_a": name_a,
        "split_b": name_b,
        "split_a_size": len(split_a),
        "split_b_size": len(split_b),
        "cross_split_near_duplicates": len(cross_leaks),
        "threshold": threshold,
        "leaks": cross_leaks[:20],
    }


def make_provenance(episodes: list[dict], result: dict) -> dict:
    from auditkit import sha256_digest
    from auditkit.provenance import make_provenance, input_ref

    corpus_hash = sha256_digest(
        json.dumps([ep["episode_id"] for ep in episodes], sort_keys=True)
    )
    return make_provenance(
        method="curatorkit_validation",
        inputs=[
            input_ref("dataset", "data/processed/turndrift_core.jsonl"),
            input_ref("schema", "data/schema.json"),
            input_ref("calibration", "data/calibration/"),
        ],
        params={
            "corpus_sha256": corpus_hash,
            "schema_gate_min_tokens": 20,
            "minhash_threshold": 0.85,
            "n_episodes": len(episodes),
            "library": "curatorkit",
            "library_version": __import__("curatorkit").__version__,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/curatorkit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_path = Path("data/processed/turndrift_core.jsonl")
    schema_path = Path("data/schema.json")
    cal_dir = Path("data/calibration")

    episodes = load_episodes(core_path)
    samples = [episode_to_sample(ep) for ep in episodes]
    dev_samples = [s for s in samples if s.metadata["split"] == "dev"]
    test_samples = [s for s in samples if s.metadata["split"] == "test"]

    print(f"Loaded {len(episodes)} episodes ({len(dev_samples)} dev, {len(test_samples)} test)")

    # 1. JSON Schema validation
    print("\n1. JSON Schema validation...")
    schema_result = validate_schema(episodes, schema_path)
    print(f"   Valid: {schema_result['valid']}/{schema_result['total']}")

    # 2. CuratorKIT SchemaGate
    print("\n2. CuratorKIT SchemaGate...")
    gate_result = run_schema_gate(samples)
    print(f"   Passed: {gate_result['after']}/{gate_result['before']}")

    # 3. Exact deduplication
    print("\n3. Exact deduplication...")
    dedup_result = run_exact_dedup(episodes)
    print(f"   Unique: {dedup_result['unique']}/{dedup_result['total']}")

    # 4. MinHash near-duplicate detection (within full dataset)
    print("\n4. MinHash near-duplicate detection (full dataset)...")
    minhash_result = run_minhash_dedup(samples, threshold=0.85)
    print(f"   After dedup: {minhash_result['after']}/{minhash_result['before']} (removed {minhash_result['near_duplicates_removed']})")

    # 5. Cross-split leak checks
    print("\n5. Cross-split leak checks...")
    dev_test_leak = cross_split_leak_check(dev_samples, test_samples, "dev", "test")
    print(f"   Dev vs Test near-duplicates: {dev_test_leak['cross_split_near_duplicates']}")

    # 6. Calibration vs test leak check
    cal_samples = []
    for cal_file in sorted(cal_dir.glob("*.jsonl")):
        with open(cal_file) as f:
            for line in f:
                entry = json.loads(line)
                cal_id = f"cal_{cal_file.stem}_{len(cal_samples)}"
                cal_samples.append(DataSample(
                    source_uri=f"calibration::{cal_id}",
                    instruction=entry.get("system_message", entry.get("text", "")),
                    output=entry.get("text", json.dumps(entry.get("turns", []))),
                    metadata={"source": cal_file.stem, "episode_id": cal_id},
                ))

    print(f"   Loaded {len(cal_samples)} calibration entries")
    cal_test_leak = cross_split_leak_check(cal_samples, test_samples, "calibration", "test")
    print(f"   Calibration vs Test near-duplicates: {cal_test_leak['cross_split_near_duplicates']}")

    cal_dev_leak = cross_split_leak_check(cal_samples, dev_samples, "calibration", "dev")
    print(f"   Calibration vs Dev near-duplicates: {cal_dev_leak['cross_split_near_duplicates']}")

    # Provenance
    provenance = make_provenance(episodes, {})

    # Assemble report
    report = {
        "stage": "curatorkit_validation",
        "library": "curatorkit",
        "library_version": __import__("curatorkit").__version__,
        "json_schema_validation": schema_result,
        "schema_gate": gate_result,
        "exact_dedup": dedup_result,
        "minhash_near_dedup": minhash_result,
        "leak_check_dev_vs_test": dev_test_leak,
        "leak_check_calibration_vs_test": cal_test_leak,
        "leak_check_calibration_vs_dev": cal_dev_leak,
        "domain_split": {
            "method": "domain-level: all episodes from a domain go to the same split",
            "dev_domains": 7,
            "test_domains": 3,
        },
        "provenance": provenance,
    }

    out_file = out_dir / "curatorkit_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_file}")

    # Summary
    print("\n" + "=" * 60)
    print("CURATORKIT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes:           {len(episodes)}")
    print(f"  Schema valid:       {schema_result['valid']}/{schema_result['total']}")
    print(f"  SchemaGate passed:  {gate_result['after']}/{gate_result['before']}")
    print(f"  Exact duplicates:   {dedup_result['duplicates']}")
    print(f"  MinHash near-dups:  {minhash_result['near_duplicates_removed']} (threshold={minhash_result['threshold']})")
    print(f"  Dev↔Test leaks:     {dev_test_leak['cross_split_near_duplicates']}")
    print(f"  Cal↔Test leaks:     {cal_test_leak['cross_split_near_duplicates']}")
    print(f"  Cal↔Dev leaks:      {cal_dev_leak['cross_split_near_duplicates']}")


if __name__ == "__main__":
    main()
