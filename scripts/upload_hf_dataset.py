#!/usr/bin/env python3
"""Upload TurnDrift-Control dataset + results to HuggingFace.

Usage:
    # First login:
    python -c "from huggingface_hub import login; login()"

    # Then upload:
    python scripts/upload_hf_dataset.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "nirdesho6o/turndrift-control"
REPO_TYPE = "dataset"

api = HfApi()


def main():
    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)

    uploads = []

    # Dataset episodes
    data_dir = Path("data/processed")
    if data_dir.exists():
        for f in sorted(data_dir.glob("*.jsonl")):
            uploads.append((str(f), f"data/{f.name}"))

    # Schema
    schema = Path("data/schema.json")
    if schema.exists():
        uploads.append((str(schema), "data/schema.json"))

    # Splits
    splits_dir = Path("data/splits")
    if splits_dir.exists():
        for f in sorted(splits_dir.glob("*.json")):
            uploads.append((str(f), f"data/splits/{f.name}"))

    # Calibration sets
    cal_dir = Path("data/calibration")
    if cal_dir.exists():
        for f in sorted(cal_dir.glob("*.jsonl")):
            uploads.append((str(f), f"data/calibration/{f.name}"))

    # Base transcripts
    base_dir = Path("results/transcripts/base")
    if base_dir.exists():
        for f in sorted(base_dir.glob("*.json")):
            uploads.append((str(f), f"transcripts/base/{f.name}"))

    # Monitor results
    for mode in ["base", "base_per_turn"]:
        mon_dir = Path(f"results/monitor/{mode}")
        if mon_dir.exists():
            for f in sorted(mon_dir.glob("*.json")):
                uploads.append((str(f), f"monitor/{mode}/{f.name}"))

    # Monitor summaries
    for s in ["base_summary.json", "base_per_turn_summary.json"]:
        p = Path(f"results/monitor/{s}")
        if p.exists():
            uploads.append((str(p), f"monitor/{s}"))

    # Judge results
    judge_dir = Path("results/judge")
    if judge_dir.exists():
        for f in sorted(judge_dir.rglob("*.json")):
            rel = f.relative_to(judge_dir)
            uploads.append((str(f), f"judge/{rel}"))

    # EGTA
    eval_dir = Path("results/evaluations")
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.json")):
            uploads.append((str(f), f"evaluations/{f.name}"))

    # Figures
    fig_dir = Path("results/figures")
    if fig_dir.exists():
        for f in sorted(fig_dir.glob("*.png")):
            uploads.append((str(f), f"figures/{f.name}"))

    # Pilot (D1/D2 AUROC, steering vectors)
    pilot_dir = Path("results/pilot")
    if pilot_dir.exists():
        for f in sorted(pilot_dir.glob("*.json")):
            uploads.append((str(f), f"pilot/{f.name}"))
        # D1/D2 direction vectors
        for f in sorted(pilot_dir.glob("*.pt")):
            uploads.append((str(f), f"pilot/{f.name}"))
        for f in sorted(pilot_dir.glob("*.npy")):
            uploads.append((str(f), f"pilot/{f.name}"))

    # CuratorKIT report
    ck_dir = Path("results/curatorkit")
    if ck_dir.exists():
        for f in sorted(ck_dir.glob("*.json")):
            uploads.append((str(f), f"curatorkit/{f.name}"))

    # AuditKIT audit record + provenance
    ak_dir = Path("results/auditkit")
    if ak_dir.exists():
        for f in sorted(ak_dir.glob("*.json")):
            uploads.append((str(f), f"auditkit/{f.name}"))

    # Dataset card
    card = Path("DATASET_CARD.md")
    if card.exists():
        uploads.append((str(card), "README.md"))

    print(f"Uploading {len(uploads)} files to {REPO_ID}...")
    for local, remote in uploads:
        print(f"  {local} → {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")

    # --- Model repo for D1/D2 vectors ---
    model_repo = "nirdesho6o/turndrift-d2-vectors"
    create_repo(model_repo, repo_type="model", exist_ok=True, private=False)

    model_uploads = []
    pilot_dir = Path("results/pilot")
    if pilot_dir.exists():
        for f in sorted(pilot_dir.glob("*.json")):
            model_uploads.append((str(f), f.name))
        for f in sorted(pilot_dir.glob("*.pt")):
            model_uploads.append((str(f), f.name))
        for f in sorted(pilot_dir.glob("*.npy")):
            model_uploads.append((str(f), f.name))

    model_card = Path("D2_MODEL_CARD.md")
    if model_card.exists():
        model_uploads.append((str(model_card), "README.md"))

    print(f"\nUploading {len(model_uploads)} files to {model_repo}...")
    for local, remote in model_uploads:
        print(f"  {local} → {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=model_repo,
            repo_type="model",
        )

    print(f"\nDone! https://huggingface.co/nirdesho6o/turndrift-d2-vectors")


if __name__ == "__main__":
    main()
