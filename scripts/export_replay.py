#!/usr/bin/env python3
"""Export merged transcripts for the replay Space.

Combines episode definitions (data/processed), model responses
(results/transcripts/base), and monitor verdicts into a single
JSON per episode for the Gradio replay viewer.

Usage:
    python scripts/export_replay.py --out spaces/transcripts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="spaces/transcripts")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    episodes = {}
    data_dir = Path(args.data_dir)
    for f in sorted(data_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                ep = json.loads(line)
                episodes[ep["episode_id"]] = ep

    # Load responses
    results_dir = Path(args.results_dir)
    trans_dir = results_dir / "transcripts" / "base"
    responses = {}
    if trans_dir.exists():
        for f in sorted(trans_dir.glob("*.json")):
            if f.stem.startswith("_"):
                continue
            with open(f) as fh:
                r = json.load(fh)
            responses[r["episode_id"]] = r

    # Load monitor verdicts
    monitors = {}
    for mode, key in [("base", "monitor_full_context"), ("base_per_turn", "monitor_per_turn")]:
        mon_dir = results_dir / "monitor" / mode
        if not mon_dir.exists():
            continue
        for f in sorted(mon_dir.glob("*.json")):
            with open(f) as fh:
                v = json.load(fh)
            eid = v["episode_id"]
            if eid not in monitors:
                monitors[eid] = {}
            monitors[eid][key] = {
                "verdict": v.get("monitor_verdict", v.get("verdict")),
                "first_block_turn": v.get("first_block_turn"),
            }

    count = 0
    for eid, ep in sorted(episodes.items()):
        merged = dict(ep)

        # Add responses
        if eid in responses:
            merged["responses"] = responses[eid].get("responses", [])
            merged["compliance"] = responses[eid].get("compliance", {})

        # Add monitor verdicts
        if eid in monitors:
            merged.update(monitors[eid])

        out_file = out_dir / f"{eid}.json"
        with open(out_file, "w") as fh:
            json.dump(merged, fh, indent=2)
        count += 1

    print(f"Exported {count} episodes to {out_dir}/")


if __name__ == "__main__":
    main()
