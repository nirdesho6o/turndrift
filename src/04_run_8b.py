"""8B model runner via ZeroGPU Space (nirdesho6o/aya8b-check).

Calls the Space's /run_episode API endpoint for each episode.
Caches transcripts so quota cutoffs just resume.

Usage:
    python src/04_run_8b.py
    python src/04_run_8b.py --split dev --limit 10
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SPACE_ID = "nirdesho6o/aya8b-check"
DATASET_PATH = Path("data/processed/turndrift_core.jsonl")
OUTPUT_DIR = Path("results/transcripts/base_8b")


def load_episodes(path: str = str(DATASET_PATH)) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Run 8B model via ZeroGPU Space")
    parser.add_argument("--space", default=SPACE_ID, help="HF Space ID")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None, choices=["dev", "test"])
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    from gradio_client import Client

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set in environment")
        sys.exit(1)

    print(f"Connecting to Space {args.space}...")
    client = Client(args.space, hf_token=hf_token)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.dataset)
    if args.split:
        episodes = [e for e in episodes if e["split"] == args.split]
    if args.limit:
        episodes = episodes[:args.limit]

    # Skip existing (resume after quota cutoff)
    existing = {p.stem for p in out_dir.glob("*.json")}
    before = len(episodes)
    episodes = [e for e in episodes if e["episode_id"] not in existing]
    if before != len(episodes):
        print(f"Skipping {before - len(episodes)} existing transcripts")

    if not episodes:
        print("No episodes to run.")
        return

    print(f"Running {len(episodes)} episodes via {args.space}")

    compliant = 0
    total_attacks = 0
    results_summary = []
    quota_exhausted = False

    for i, episode in enumerate(episodes):
        eid = episode["episode_id"]
        print(f"[{i+1}/{len(episodes)}] {eid} ", end="", flush=True)

        transcript = None
        for attempt in range(args.max_retries):
            try:
                result = client.predict(
                    episode_json=json.dumps(episode),
                    api_name="/run_episode",
                )
                transcript = json.loads(result)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "gpu" in err_str and "exceeded" in err_str:
                    print(f"\nGPU quota exhausted after {i} episodes. Resume later.")
                    quota_exhausted = True
                    break
                wait = min(2 ** (attempt + 1), 60)
                print(f"\n  retry {attempt+1}/{args.max_retries} in {wait}s: {e}")
                time.sleep(wait)

        if quota_exhausted:
            break

        if transcript is None:
            print(f"FAILED after {args.max_retries} retries")
            continue

        out_path = out_dir / f"{eid}.json"
        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2)

        if episode["variant"] == "attack":
            total_attacks += 1
            if transcript.get("compliance", {}).get("detected"):
                compliant += 1
                print(f"COMPLIANT (turn {transcript['compliance']['after_turn']})")
            else:
                print("REFUSED")
        else:
            print("BENIGN")

        results_summary.append({
            "episode_id": eid,
            "variant": episode["variant"],
            "strategy": episode.get("injection_strategy", "none"),
            "target": episode.get("injection_target", "none"),
            "compliant": (transcript["compliance"]["detected"]
                          if transcript.get("compliance") else None),
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"8B MODEL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes completed: {len(results_summary)}")
    if quota_exhausted:
        print("NOTE: Run incomplete due to GPU quota. Re-run to resume.")

    if total_attacks > 0:
        icr = compliant / total_attacks * 100
        print(f"Attack episodes: {total_attacks}")
        print(f"Injection Compliance Rate: {compliant}/{total_attacks} ({icr:.1f}%)")

        print(f"\nBy strategy:")
        strategies = set(r["strategy"] for r in results_summary if r["variant"] == "attack")
        for strat in sorted(strategies):
            sr = [r for r in results_summary if r["strategy"] == strat and r["variant"] == "attack"]
            sc = sum(1 for r in sr if r["compliant"])
            print(f"  {strat}: {sc}/{len(sr)} ({100*sc/len(sr):.0f}%)")

        print(f"\nBy target:")
        targets = set(r["target"] for r in results_summary if r["variant"] == "attack")
        for tgt in sorted(targets):
            tr = [r for r in results_summary if r["target"] == tgt and r["variant"] == "attack"]
            tc = sum(1 for r in tr if r["compliant"])
            print(f"  {tgt}: {tc}/{len(tr)} ({100*tc/len(tr):.0f}%)")

    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "CohereLabs/aya-expanse-8b",
            "space": args.space,
            "total_episodes": len(results_summary),
            "total_attacks": total_attacks,
            "compliant": compliant,
            "icr": compliant / total_attacks if total_attacks > 0 else 0,
            "quota_exhausted": quota_exhausted,
            "results": results_summary,
        }, f, indent=2)

    print(f"\nTranscripts saved to {out_dir}/")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
