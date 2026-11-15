# TurnDrift-Control — Colab Setup

Two cells. GPU runtime required (T4 minimum).

## Cell 1: Setup

```python
# Mount Drive for persistent results
from google.colab import drive
drive.mount('/content/drive')

# Set tokens from Colab secrets
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["ADAPTION_API_KEY"] = userdata.get("ADAPTION_API_KEY")
GH_TOKEN = userdata.get("GH_TOKEN")

# Clone or pull private repo (needs GH_TOKEN with repo scope)
import subprocess
repo_dir = "/content/lexsi"
clone_url = f"https://{GH_TOKEN}@github.com/nirdesho6o/turndrift.git"
if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", clone_url, repo_dir], check=True)
else:
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

# Install dependencies (includes adaption SDK for over-refusal judging)
subprocess.run(["pip", "install", "-q", "-r", f"{repo_dir}/requirements.txt"], check=True)
```

## Cell 2: Run

```python
!python /content/lexsi/scripts/colab_run.py \
    --results-dir /content/drive/MyDrive/turndrift_results \
    --batch-size 16
```

To resume after a disconnect, just re-run both cells — all completed stages are skipped.

To run specific stages: `--stages pilot_a1,gate_check`

To override a kill rule: `--override-gate`

Batch size: default 16 (T4). Auto-reduces on OOM. Batched-vs-unbatched equivalence is verified on 3 episodes at the start of `full_base` and logged to `results/batch_equivalence.json`. All conditions (base, D1, D2, random) use the same batch config; mixing configs fails loudly. Steering sanity check runs before each steered condition and is logged to `results/steering_sanity.json`.
