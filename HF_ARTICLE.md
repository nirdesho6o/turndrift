# When Safety Looks the Wrong Way: Multi-Turn Prompt Injection as an Information-Hiding Game

*Lexsi Labs AI Safety & Alignment Hackathon — Track 2a.3*

## TL;DR

We built TurnDrift-Control, a benchmark of 120 scripted agent conversations that tests whether CohereLabs/tiny-aya-global (3.35B) follows hidden instructions planted across multiple turns. Three findings:

1. **Defense shifts the attacker's best response.** Without monitoring, single-turn injection succeeds 37.5% of the time (dev). A full-context monitor (Qwen 3.5 397B) forces the attacker to switch strategy, reducing worst-case ASR to 16.7%.

2. **The full-context monitor catches single-turn injections before compliance (7/7) but mostly fires too late for split payloads (2/7).** With a 10% over-refusal budget, the LP-optimal policy puts 54% weight on the full-context monitor, 43% on no defense, and 3% on per-turn (worst-case ASR: 16.2%).

3. **Split-payload prediction falsified.** We predicted that splitting injections across turns would maintain compliance. Instead, multi-fragment strategies had 0.32x the single-turn compliance rate. Tiny Aya cannot reliably assemble fragmented instructions at 3.35B scale.

4. **Pre-injection leakage.** Two customer-support canary-extraction episodes were reclassified: the model hallucinated the canary token at turn 1, before any injection fragment. Benign twins produced the same token.

## What we did

- **120 episodes** (60 attack + 60 benign twins), 5 injection strategies, 7 proxy targets
- **Deterministic detection** — no LLM judges success/failure
- **Game-theoretic analysis** (EGTA) with LP-optimal mixed defender strategy
- **Activation steering** — extracted D1 (refusal) and D2 (injection-contrast) direction vectors; D2 achieves AUROC 0.946 at layer 16, confirming linear separability, but GPU steering was not tested (Colab T4 quota exhausted)

## What we did NOT complete

We are honest about this:

- **Activation steering** was designed but not tested on GPU (coefficient scaling issue + compute constraint)
- **8B replication** was cut for time
- **Statistical tests** require steering condition data we don't have
- **Base transcripts** have a known `position_ids` bug (batch_size=16, no cumulative sum) — compliance detection is unaffected (deterministic) but text quality may differ

## Results at a glance

| Metric | Dev (n=45 attacks) | Test (n=15 attacks) |
|--------|-------------------|---------------------|
| Base ICR (single-turn) | 37.5% (3/8) | 100.0% (4/4) |
| Base ICR (overall) | 17.8% (8/45) | 40.0% (6/15) |
| Full-ctx monitor detection | 82.2% (37/45) | 80.0% (12/15) |
| Full-ctx monitor FPR | 13.3% (8/60) | 13.3% (8/60) |
| Per-turn monitor FPR | 93.3% (56/60) | 93.3% (56/60) |
| LP-optimal worst-case ASR | 16.2% | 41.6% |
| Blocked in time (single) | 7/7 (100%) | — |
| Blocked in time (split) | 2/7 (28.6%) | — |

Two episodes were reclassified as pre-injection leakage (excluded from ICR). The per-turn monitor catches everything (100%) but flags 93.3% of benign conversations — unusable in practice.

## Toolkits

- **CuratorKIT**: dataset validation and provenance
- **SafeTune**: D1/D2 direction extraction, steering hook
- **AuditKIT**: evaluation recording with model fingerprints

## Links

- **Code**: [github.com/nirdesho6o/turndrift](https://github.com/nirdesho6o/turndrift)
- **Dataset**: [huggingface.co/datasets/nirdesho6o/turndrift-control](https://huggingface.co/datasets/nirdesho6o/turndrift-control)
- **Demo**: [huggingface.co/spaces/nirdesho6o/turndrift-replay](https://huggingface.co/spaces/nirdesho6o/turndrift-replay)
- **Report**: `report/report.tex` in the repo

## Author

Nirdesh Bhandari
