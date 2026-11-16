# TurnDrift-Control

**Multi-turn prompt injection benchmark for agent conversations.**

**Claim:** A full-context external monitor reduces worst-case attack success against a 3.35B agent model from 37.5% to 16.2% (dev), but intercepts split-payload injections only after the model has already complied.

Built for the [Lexsi Labs AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai), Track 2a, Problem Statement 3 (multi-turn audits).

> **Content warning:** Proxy injection targets only. All tools mocked, nothing executes. No harmful content.

## What this is

TurnDrift-Control tests whether `CohereLabs/tiny-aya-global` (3.35B) follows hidden instructions planted across multiple turns in agent conversations, and whether external monitors can detect it.

- **120 episodes** (60 attack + 60 benign twins), 720 turns
- **5 injection strategies:** single-turn, two-fragment, three-fragment, semantic split, delayed trigger
- **7 proxy targets** (deterministic detection, nothing executes)
- **3 defense conditions:** no defense, full-context monitor, per-turn monitor
- **Game-theoretic analysis** (EGTA) of attacker vs. defender strategies

## Headline before → after

| Metric | Undefended (before) | Mixed monitor policy (after) |
|--------|-------------------|------------------------------|
| Worst-case ASR (dev) | 37.5% (n=8) | **16.2%** (LP mixture) |
| Worst-case ASR (test) | 100.0% (n=4) | **41.6%** (LP mixture) |
| Over-refusal (dev) | 0.0% | 10.0% |

## Key findings

| Metric | Dev | Test |
|--------|-----|------|
| Base ICR (single-turn) | 37.5% (3/8) | 100.0% (4/4) |
| Base ICR (overall) | 17.8% (8/45) | 40.0% (6/15) |
| Full-context monitor detection (all attacks) | 82.2% (37/45) | 80.0% (12/15) |
| Full-context monitor FPR | 13.3% (8/60) | 13.3% (8/60) |
| LP-optimal worst-case ASR | 16.2% | 41.6% |
| Compliance ratio (multi-frag / single) | 0.32x | 0.17x |
| Blocked in time: single-turn | 7/7 (100%) | — |
| Blocked in time: split | 2/7 (28.6%) | — |

**Pre-injection secret leakage:** The model spontaneously leaks the system-prompt canary in 10% of benign episodes (6/60), all in the `customer_support` domain (6/6, 100%). Two canary-extraction episodes excluded from ICR/EGTA. The full-context monitor intercepts single-turn injections before compliance (7/7), but for split payloads it mostly fires only after the model has already complied (2/7). Splitting injections across turns made the model *less* likely to comply — **falsifying** our pre-registered prediction.

## Kits used

- **CuratorKIT** — schema validation (120/120), exact dedup (70 unique / 120), embedding near-dup detection, cross-split leak checks
- **SafeTune** — D1/D2 direction extraction (D2 AUROC 0.946 at layer 16), steering hook (untested on GPU)
- **AuditKIT** — evaluation recording with provenance chain and model fingerprints

## Repository structure

```
data/               Benchmark episodes (JSONL) and calibration sets
scripts/            Pipeline scripts (generation, inference, monitoring, evaluation)
src/turndrift/      Core library (detectors, generators, episode schema)
tests/              102 tests
report/             LaTeX report with figures
spaces/             Gradio demos (replay viewer)
results/            Cached outputs (gitignored — published on HuggingFace)
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_benchmark.py          # Generate benchmark
python scripts/run_monitor.py --mode full_context  # Run monitors (needs ADAPTION_API_KEY)
python scripts/run_curatorkit.py              # CuratorKIT validation
python scripts/run_auditkit.py                # AuditKIT recording
python scripts/run_egta.py --split dev        # EGTA analysis
python scripts/make_figures.py --split dev    # Figures
python -m pytest tests/ -q                    # Tests (102 pass)
```

## What was not completed

- Activation steering (coefficient scaling issue, GPU quota exhausted)
- 8B replication (compute constraint)
- Statistical tests (no steering condition data)
- Hindi multilingual factor
- Human review of 20% sample

See `report/report.tex` Section 6 and the full write-up for details.

## Artifacts

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/nirdesho6o/turndrift-control)
- **Model (D2 vectors):** [HuggingFace](https://huggingface.co/nirdesho6o/turndrift-d2-vectors)
- **Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/nirdesho6o/turndrift-replay)
- **Write-up:** [HuggingFace Article](https://huggingface.co/blog/nirdesho6o/turndrift-control)
- **Report:** `report/report.tex`

## Author

Nirdesh Bhandari
