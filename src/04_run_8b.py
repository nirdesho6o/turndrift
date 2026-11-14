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
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Episodes per ZeroGPU batch call (default 16, H200 70GB)")
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

    # Estimate ZeroGPU budget (H200 70GB)
    n_batches = (len(episodes) + args.batch_size - 1) // args.batch_size
    secs_per_single = 180
    # Duration per batch: 6 turns × ~8s/turn × batch_size/16 scaling, × 1.3 margin
    secs_per_batch = int(6 * 8 * max(args.batch_size, 1) / 16 * 1.3)
    secs_per_batch = max(secs_per_batch, 60)
    if args.batch_size == 1:
        est_gpu_secs = len(episodes) * secs_per_single
    else:
        est_gpu_secs = n_batches * secs_per_batch
    est_gpu_mins = est_gpu_secs / 60
    free_mins = 25
    paid_rate = 0.10  # $1 per 10 min on H200
    paid_cost = max(0, est_gpu_mins - free_mins) * paid_rate

    print(f"\nZeroGPU budget estimate (H200 70GB):")
    print(f"  Episodes: {len(episodes)}, batch_size={args.batch_size}, batches={n_batches}")
    print(f"  Estimated GPU time: {est_gpu_mins:.1f} min")
    print(f"  Free tier: 25 min/day")
    if est_gpu_mins <= free_mins:
        print(f"  Cost: $0 (within free tier)")
    else:
        print(f"  Estimated cost: ${paid_cost:.2f} "
              f"({est_gpu_mins - free_mins:.1f} min overage × $0.10/min)")
    print(f"  vs unbatched: {len(episodes) * secs_per_single / 60:.1f} min")
    print()

    print(f"Running {len(episodes)} episodes via {args.space} (batch_size={args.batch_size})")

    compliant = 0
    total_attacks = 0
    results_summary = []
    quota_exhausted = False

    for batch_start in range(0, len(episodes), args.batch_size):
        batch = episodes[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        batch_eids = [e["episode_id"] for e in batch]
        print(f"\nBatch {batch_num}/{n_batches} ({len(batch)} episodes): ", end="", flush=True)

        transcripts = None
        for attempt in range(args.max_retries):
            try:
                if args.batch_size > 1:
                    result = client.predict(
                        episodes_json=json.dumps(batch),
                        api_name="/run_episodes_batch",
                    )
                    transcripts = json.loads(result)
                else:
                    result = client.predict(
                        episode_json=json.dumps(batch[0]),
                        api_name="/run_episode",
                    )
                    transcripts = [json.loads(result)]
                break
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or ("gpu" in err_str and "exceeded" in err_str):
                    print(f"\nGPU quota exhausted after batch {batch_num}. Resume later.")
                    quota_exhausted = True
                    break
                wait = min(2 ** (attempt + 1), 60)
                print(f"\n  retry {attempt+1}/{args.max_retries} in {wait}s: {e}")
                time.sleep(wait)

        if quota_exhausted:
            break

        if transcripts is None:
            print(f"FAILED after {args.max_retries} retries")
            continue

        print(f"done")
        for transcript in transcripts:
            eid = transcript["episode_id"]
            out_path = out_dir / f"{eid}.json"
            with open(out_path, "w") as f:
                json.dump(transcript, f, indent=2)

            ep = next(e for e in batch if e["episode_id"] == eid)
            if ep["variant"] == "attack":
                total_attacks += 1
                if transcript.get("compliance", {}).get("detected"):
                    compliant += 1
                    print(f"  {eid}: COMPLIANT (turn {transcript['compliance']['after_turn']})")
                else:
                    print(f"  {eid}: REFUSED")
            else:
                print(f"  {eid}: BENIGN")

            results_summary.append({
                "episode_id": eid,
                "variant": ep["variant"],
                "strategy": ep.get("injection_strategy", "none"),
                "target": ep.get("injection_target", "none"),
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
