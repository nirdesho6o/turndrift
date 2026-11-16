---
license: apache-2.0
tags:
- safety
- prompt-injection
- activation-steering
- direction-vectors
library_name: safetune
pipeline_tag: feature-extraction
---

# TurnDrift-Control D1/D2 Direction Vectors

**Status: UNVALIDATED as steering vectors.** These directions achieve high classification AUROC but have not been tested for steering on GPU.

## What these are

Direction vectors extracted from `CohereLabs/tiny-aya-global` (3.35B, FP16) for:

- **D1 (refusal direction):** Mean difference between harmful and harmless prompt activations. Extracted via SafeTune `extract_refusal_direction`.
- **D2 (injection-contrast direction):** Mean difference between injection and clean conversation activations. Extracted via mean-difference at each layer with 3-fold cross-validation.

## D2 Quality

| Metric | Value |
|--------|-------|
| Best layer | 16 |
| AUROC (3-fold CV) | **0.946** |
| Calibration set | 45 injection + 45 clean |
| Method | Last-token mean-difference, per-layer |

## Calibration data hashes

| File | SHA-256 |
|------|---------|
| `data/calibration/injection.jsonl` | See `results/auditkit/audit_record.json` |
| `data/calibration/clean.jsonl` | See `results/auditkit/audit_record.json` |

## Why unvalidated

Steering was implemented (`RefusalDirectionModel`-style ablation + CAA-style addition) but:
1. Coefficients (α=1) were too small relative to residual stream norms (~hundreds)
2. Scale-relative coefficient grid was designed but GPU quota was exhausted
3. No steering evaluation was run

## Intended use

Research only. These vectors may be useful for:
- Linear probe classification of injection vs. clean inputs
- Starting point for steering experiments with norm-relative scaling
- Reproduction of the TurnDrift-Control D2 AUROC result

## Source

Extracted as part of the [TurnDrift-Control](https://github.com/nirdesho6o/turndrift) benchmark for the Lexsi Labs AI Safety & Alignment Hackathon, Track 2a.3.

## License

Apache 2.0
