# When Safety Looks the Wrong Way: Multi-Turn Prompt Injection as an Information-Hiding Game

*Lexsi Labs AI Safety & Alignment Hackathon — Track 2a.3*

## TL;DR

We built TurnDrift-Control, a benchmark of 120 scripted agent conversations that tests whether CohereLabs/tiny-aya-global (3.35B) follows hidden instructions planted across multiple turns. Three findings:

1. **Defense shifts the attacker's best response.** Without monitoring, single-turn injection succeeds 50% of the time. A full-context monitor (Qwen 3.5 397B) forces the attacker to switch to semantic-split injection, reducing worst-case ASR to 21.4%.

2. **The FPR-budgeted optimal defense is a randomized full-context monitor.** With a 10% over-refusal budget, the LP-optimal policy puts 75% weight on the full-context monitor and 25% on no defense (worst-case ASR: 21.8%).

3. **Split-payload prediction falsified.** We predicted that splitting injections across turns would maintain compliance. Instead, multi-fragment strategies had 0.27x the single-turn compliance rate. Tiny Aya cannot reliably assemble fragmented instructions at 3.35B scale.

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
| Base ICR (single-turn) | 50.0% (4/8) | 100.0% (4/4) |
| Base ICR (overall) | 22.2% (10/45) | 40.0% (6/15) |
| Full-ctx monitor detection | 77.8% (35/45) | 80.0% (12/15) |
| Full-ctx monitor FPR | 13.3% (8/60) | 13.3% (8/60) |
| Per-turn monitor FPR | 93.3% (56/60) | 93.3% (56/60) |
| LP-optimal worst-case ASR | 21.8% | 41.6% |

The per-turn monitor catches nearly everything (97.8%) but flags 93.3% of benign conversations — unusable in practice. The pre-registered per-turn innocuousness test is uninformative because the per-turn FPR is so high.

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
