---
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- prompt-injection
- ai-safety
- multi-turn
- benchmark
- game-theory
pretty_name: TurnDrift-Control
size_categories:
- n<1K
---

# TurnDrift-Control

Multi-turn prompt injection benchmark for agent conversations.

## Overview

TurnDrift-Control tests whether language models follow hidden instructions
planted across multiple turns in agent conversations. It pairs each attack
episode with a benign twin (same topic, no injection) to control for
over-refusal.

- **120 episodes** (60 attack + 60 benign twins), 720 turns
- **5 injection strategies:** single-turn, two-fragment, three-fragment, semantic split, delayed trigger
- **7 proxy injection targets** with deterministic detection (nothing executes)
- **10 domain templates**, split 7 dev / 3 test by domain

## Key Results (CohereLabs/tiny-aya-global, 3.35B)

| Metric | Dev | Test |
|--------|-----|------|
| Base ICR (overall) | 22.2% (10/45) | 40.0% (6/15) |
| Base ICR (single-turn) | 50.0% (4/8) | 100.0% (4/4) |
| Full-context monitor detection (all attacks) | 77.8% (35/45) | 80.0% (12/15) |
| Full-context monitor FPR | 13.3% (8/60) | 13.3% (8/60) |
| LP-optimal worst-case ASR | 21.8% | 41.6% |
| Fragment compliance ratio | 0.27x | 0.17x |

**Headline:** Defense shifts the attacker's best response from single-turn
to semantic-split injection. The FPR-budgeted optimal defense is a
randomized full-context monitor (75.2% weight, 10% over-refusal budget).

**Falsified prediction:** Split payloads reduce compliance (0.27x),
not maintain it as predicted.

## Provisional Base Caveat

Base transcripts were generated with a `position_ids` bug (batch_size=16,
no cumulative sum). Compliance detection is deterministic and unaffected,
but generated text quality may differ from a correct implementation.

## Not Completed

- Activation steering (D2 AUROC = 0.946 at layer 16, but GPU steering untested)
- 8B replication
- Statistical tests (no steering condition data)
- Hindi multilingual factor

## Structure

```
data/               Episode JSONL files + calibration sets
transcripts/base/   Model transcripts with compliance labels
monitor/            Full-context and per-turn monitor verdicts
judge/              Benign helpfulness labels + kappa report
evaluations/        EGTA analysis (dev + test)
figures/            Publication figures
pilot/              D1/D2 direction vectors + AUROC summary
```

## Citation

```bibtex
@misc{bhandari2026turndrift,
  title={When Safety Looks the Wrong Way: Multi-Turn Prompt Injection as an Information-Hiding Game},
  author={Bhandari, Nirdesh},
  year={2026},
  howpublished={Lexsi Labs AI Safety \& Alignment Hackathon, Track 2a.3},
  url={https://github.com/nirdesho6o/turndrift}
}
```

## License

Apache 2.0
