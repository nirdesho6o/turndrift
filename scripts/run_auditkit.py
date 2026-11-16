#!/usr/bin/env python3
"""AuditKIT evaluation recording with model fingerprints for TurnDrift-Control.

Records the evaluation runs (base compliance, monitor detection, EGTA defense
comparison) with provenance and model fingerprints. Chains with CuratorKIT
output (references the curatorkit_report.json as input provenance).

Usage:
    python scripts/run_auditkit.py [--out results/auditkit]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from auditkit import sha256_digest
from auditkit.provenance import make_provenance, input_ref, write_provenance


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint(model_id: str, revision: str | None = None) -> dict:
    return {
        "model_id": model_id,
        "revision": revision,
        "fingerprint": sha256_digest(f"{model_id}@{revision or 'main'}"),
    }


def record_base_eval(results_dir: Path) -> dict:
    transcripts_dir = results_dir / "transcripts" / "base"
    episodes = []
    compliant = 0
    excluded = 0
    for f in sorted(transcripts_dir.glob("atk_*.json")):
        if f.stem.startswith("_"):
            continue
        t = json.load(open(f))
        c = t.get("compliance", {})
        if c.get("pre_injection_leakage"):
            excluded += 1
            continue
        episodes.append(t["episode_id"])
        if c.get("detected"):
            compliant += 1

    return {
        "run_id": "base_compliance_v1",
        "method": "deterministic_detector",
        "model": model_fingerprint(
            "CohereLabs/tiny-aya-global",
            revision="main",
        ),
        "n_episodes": len(episodes),
        "n_excluded": excluded,
        "exclusion_reason": "pre-injection canary leakage in customer_support domain",
        "compliant": compliant,
        "icr": round(compliant / len(episodes) * 100, 1) if episodes else 0,
        "corpus_fingerprint": sha256_digest(json.dumps(sorted(episodes))),
    }


def record_monitor_eval(results_dir: Path, mode: str) -> dict:
    if mode == "full_context":
        mon_dir = results_dir / "monitor" / "base"
    else:
        mon_dir = results_dir / "monitor" / "base_per_turn"

    if not mon_dir.exists():
        return {"error": f"Monitor directory not found: {mon_dir}"}

    total = 0
    blocked = 0
    benign_blocked = 0
    benign_total = 0
    for f in sorted(mon_dir.glob("*.json")):
        v = json.load(open(f))
        total += 1
        eid = v["episode_id"]
        is_benign = eid.startswith("ben_")
        if is_benign:
            benign_total += 1
        fb = v.get("first_block_turn")
        if fb is not None:
            blocked += 1
            if is_benign:
                benign_blocked += 1

    return {
        "run_id": f"monitor_{mode}_v1",
        "method": f"{mode}_monitor",
        "monitor_model": model_fingerprint("Qwen/Qwen3.5-397B-A39B"),
        "n_episodes": total,
        "blocked": blocked,
        "benign_blocked": benign_blocked,
        "benign_total": benign_total,
        "fpr": round(benign_blocked / benign_total * 100, 1) if benign_total else 0,
    }


def record_judge_eval(results_dir: Path) -> dict:
    judge_dir = results_dir / "judge"
    if not judge_dir.exists():
        return {"error": "Judge directory not found"}

    total = 0
    helpful = 0
    for subdir in sorted(judge_dir.glob("over_refusal/*")):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.glob("ben_*.json")):
            j = json.load(open(f))
            total += 1
            if j.get("label") == "helpful":
                helpful += 1

    kappa_file = judge_dir / "kappa_report.json"
    kappa = json.load(open(kappa_file)) if kappa_file.exists() else {}

    return {
        "run_id": "judge_v1",
        "method": "llm_judge",
        "primary_judge": model_fingerprint("Qwen/Qwen3.5-397B-A39B"),
        "secondary_judge": model_fingerprint("THUDM/GLM-5.2"),
        "n_episodes": total,
        "helpful": helpful,
        "over_refusal": total - helpful,
        "inter_judge_agreement": kappa.get("agreement_pct"),
        "cohens_kappa": kappa.get("cohens_kappa"),
        "kappa_note": "degenerate (near-zero label variance)",
    }


def record_egta(results_dir: Path) -> dict:
    egta_results = {}
    for split in ["dev", "test"]:
        f = results_dir / "evaluations" / f"egta_{split}.json"
        if f.exists():
            egta_results[split] = json.load(open(f))
    return {
        "run_id": "egta_v1",
        "method": "egta_lp_stackelberg",
        "splits": list(egta_results.keys()),
        "dev_lp_value": egta_results.get("dev", {}).get("mixed_strategy_lp", {}).get("value"),
        "test_lp_value": egta_results.get("test", {}).get("mixed_strategy_lp", {}).get("value"),
    }


def record_d2_eval(results_dir: Path) -> dict:
    a4 = results_dir / "pilot" / "a4_summary.json"
    if not a4.exists():
        return {"error": "D2 summary not found"}
    summary = json.load(open(a4))
    return {
        "run_id": "d2_direction_v1",
        "method": "safetune_mean_difference_cv",
        "model": model_fingerprint("CohereLabs/tiny-aya-global"),
        "best_layer": summary.get("best_layer"),
        "best_auroc": summary.get("best_auroc"),
        "n_folds": summary.get("n_folds"),
        "calibration_hashes": {
            "injection": file_sha256(Path("data/calibration/injection.jsonl")),
            "clean": file_sha256(Path("data/calibration/clean.jsonl")),
        },
        "status": "unvalidated_as_steering_vector",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/auditkit")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    print("Recording evaluation runs with AuditKIT...\n")

    # Record each evaluation
    base = record_base_eval(results_dir)
    print(f"Base compliance: ICR={base['icr']}% ({base['compliant']}/{base['n_episodes']}), excluded={base['n_excluded']}")

    fc_monitor = record_monitor_eval(results_dir, "full_context")
    print(f"Full-context monitor: blocked={fc_monitor.get('blocked')}, FPR={fc_monitor.get('fpr')}%")

    pt_monitor = record_monitor_eval(results_dir, "per_turn")
    print(f"Per-turn monitor: blocked={pt_monitor.get('blocked')}, FPR={pt_monitor.get('fpr')}%")

    judge = record_judge_eval(results_dir)
    print(f"Judge: helpful={judge.get('helpful')}/{judge.get('n_episodes')}, agreement={judge.get('inter_judge_agreement')}")

    egta = record_egta(results_dir)
    print(f"EGTA: dev LP value={egta.get('dev_lp_value')}, test={egta.get('test_lp_value')}")

    d2 = record_d2_eval(results_dir)
    print(f"D2 direction: layer={d2.get('best_layer')}, AUROC={d2.get('best_auroc')}")

    # CuratorKIT chain reference
    curatorkit_report = results_dir / "curatorkit" / "curatorkit_report.json"
    curatorkit_ref = None
    if curatorkit_report.exists():
        curatorkit_ref = file_sha256(curatorkit_report)
        print(f"\nChained from CuratorKIT report: {curatorkit_ref[:16]}...")

    # Build provenance
    inputs = [
        input_ref("dataset", "data/processed/turndrift_core.jsonl"),
        input_ref("transcripts", "results/transcripts/base/"),
        input_ref("monitor_full_context", "results/monitor/base/"),
        input_ref("monitor_per_turn", "results/monitor/base_per_turn/"),
        input_ref("judge", "results/judge/"),
        input_ref("egta", "results/evaluations/"),
        input_ref("d2_summary", "results/pilot/a4_summary.json"),
    ]
    if curatorkit_ref:
        inputs.append(input_ref("curatorkit_report", str(curatorkit_report), config=curatorkit_ref))

    prov = make_provenance(
        method="auditkit_evaluation_record",
        base_model="CohereLabs/tiny-aya-global",
        inputs=inputs,
        params={
            "monitor_model": "Qwen/Qwen3.5-397B-A39B",
            "judge_models": ["Qwen/Qwen3.5-397B-A39B", "THUDM/GLM-5.2"],
            "library": "auditkit",
            "library_version": __import__("auditkit").__version__,
        },
    )

    # Write provenance file
    prov_path = write_provenance(str(out_dir), prov)
    print(f"Provenance written to {prov_path}")

    # Assemble full audit record
    record = {
        "audit_id": f"turndrift_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "library": "auditkit",
        "library_version": __import__("auditkit").__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluations": {
            "base_compliance": base,
            "full_context_monitor": fc_monitor,
            "per_turn_monitor": pt_monitor,
            "judge_quality": judge,
            "egta_defense": egta,
            "d2_direction": d2,
        },
        "model_fingerprints": {
            "target": model_fingerprint("CohereLabs/tiny-aya-global"),
            "monitor": model_fingerprint("Qwen/Qwen3.5-397B-A39B"),
            "judge_primary": model_fingerprint("Qwen/Qwen3.5-397B-A39B"),
            "judge_secondary": model_fingerprint("THUDM/GLM-5.2"),
        },
        "curatorkit_chain": {
            "report_path": str(curatorkit_report) if curatorkit_ref else None,
            "report_sha256": curatorkit_ref,
        },
        "provenance": prov,
    }

    out_file = out_dir / "audit_record.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nAudit record saved to {out_file}")


if __name__ == "__main__":
    main()
