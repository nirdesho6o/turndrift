# TurnDrift-Control — Colab Setup

Two cells. GPU runtime required (T4 minimum).

## Cell 1: Setup

```python
# Mount Drive for persistent results
from google.colab import drive
drive.mount('/content/drive')

# Set HF token from Colab secrets
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# Clone or pull repo
import subprocess
repo_dir = "/content/lexsi"
if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "https://github.com/nirdesho6o/lexsi.git", repo_dir], check=True)
else:
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

# Install dependencies
subprocess.run(["pip", "install", "-q", "-r", f"{repo_dir}/requirements.txt"], check=True)
```

## Cell 2: Run

```python
!python /content/lexsi/scripts/colab_run.py --results-dir /content/drive/MyDrive/turndrift_results
```

To resume after a disconnect, just re-run both cells — all completed stages are skipped.

To run specific stages: `--stages pilot_a1,gate_check`

To override a kill rule: `--override-gate`
