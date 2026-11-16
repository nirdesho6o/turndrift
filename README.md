# TurnDrift-Control

**Multi-turn prompt injection benchmark for agent conversations.**

Built for the [Lexsi Labs AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai), Track 2a.3 (multi-turn audits).

## What this is

TurnDrift-Control tests whether a small open-weight model (`CohereLabs/tiny-aya-global`, 3.35B) follows hidden instructions planted across multiple turns in agent conversations, and whether external monitors can detect it.

- **120 episodes** (60 attack + 60 benign twins), 720 turns
- **5 injection strategies:** single-turn, two-fragment, three-fragment, semantic split, delayed trigger
- **7 proxy targets** (deterministic detection, nothing executes)
- **3 defense conditions:** no defense, full-context monitor, per-turn monitor
- **Game-theoretic analysis** (EGTA) of attacker vs. defender strategies

## Key findings

| Metric | Dev | Test |
|--------|-----|------|
| Base ICR (single-turn) | 50.0% (4/8) | 100.0% (4/4) |
| Base ICR (overall) | 22.2% (10/45) | 40.0% (6/15) |
| Full-context monitor detection (all attacks) | 77.8% (35/45) | 80.0% (12/15) |
| Full-context monitor FPR | 13.3% (8/60) | 13.3% (8/60) |
| LP-optimal worst-case ASR | 21.8% | 41.6% |
| Monitor manipulability | 16.1 pp | 33.3 pp |
| Compliance ratio (multi-frag / single) | 0.27x | 0.17x |

Defense shifts the attacker's best response from single-turn to semantic-split injection. The FPR-budgeted optimal defense is a randomized full-context monitor (75.2% weight, 10% over-refusal budget). Splitting injections across turns made the model *less* likely to comply — **falsifying** our pre-registered prediction.

## Repository structure

```
data/               Benchmark episodes (JSONL) and calibration sets
scripts/            Pipeline scripts (generation, inference, monitoring, evaluation)
src/turndrift/      Core library (detectors, generators, episode schema)
tests/              102 tests
report/             LaTeX report with figures
spaces/             Gradio demos (ZeroGPU 8B Space, CPU replay viewer)
results/            Cached outputs (gitignored — published on HuggingFace)
```

## Toolkits used

- **CuratorKIT** — dataset schema validation, deduplication, provenance
- **SafeTune** — D1/D2 direction extraction (AUROC 0.946 at layer 16), steering hook
- **AuditKIT** — evaluation recording with model fingerprints

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate benchmark
python scripts/generate_benchmark.py

# Run base inference (requires GPU)
python scripts/run_base.py --model CohereLabs/tiny-aya-global

# Run monitors (requires ADAPTION_API_KEY)
python scripts/run_monitor.py --mode full_context
python scripts/run_monitor.py --mode per_turn

# Run judge
python scripts/run_judge.py

# Run EGTA analysis
python scripts/run_egta.py --split dev
python scripts/run_egta.py --split test

# Generate figures
python scripts/make_figures.py --split dev
python scripts/make_figures.py --split test

# Run tests
python -m pytest tests/ -q
```

## What was not completed

- Activation steering (coefficient scaling issue, GPU quota exhausted)
- 8B replication (compute constraint)
- Statistical tests (no steering condition data)
- Hindi multilingual factor
- Human review of 20% sample

See `report/report.tex` Section 6 for details.

## Artifacts

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/nirdesho6o/turndrift-control)
- **Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/nirdesho6o/turndrift-replay)
- **Report:** `report/report.tex`

## Author

Nirdesh Bhandari
