# TurnDrift-Control — Submission

## Abstract

TurnDrift-Control is a benchmark of 120 scripted agent conversations (60 attack + 60 benign twins, 720 turns) testing whether CohereLabs/tiny-aya-global (3.35B parameters) follows hidden instructions planted across multiple turns. We evaluate 5 injection strategies (single-turn, two-fragment, three-fragment, semantic split, delayed trigger) against 3 defense conditions (no defense, full-context monitor, per-turn monitor) and compute the LP-optimal mixed defense via EGTA.

Without defense, the model follows hidden instructions in 17.8% of attack episodes (8/45 dev), with single-turn attacks succeeding 37.5%. A full-context monitor (Qwen 3.5 397B) reduces worst-case ASR from 37.5% to 16.7% (pure) or 18.8% (LP mixture with 10% over-refusal budget). The monitor blocks 7/7 single-turn compliant episodes before compliance but only 2/7 split-strategy episodes — for split payloads, detection fires too late. FPR is 20.0% (2/10 unique benign conversations); benign twins are domain-matched (10 unique of 60 total).

Splitting injections across turns reduced compliance to 0.32× the single-turn rate, falsifying our pre-registered prediction. An independent finding: the model spontaneously reproduces the system-prompt canary token in 1 of 10 unique benign conversations (the `customer_support` domain), leaking secrets through role-play hallucination without adversarial input. Activation steering direction vectors (D2 AUROC 0.946 at layer 16) were extracted but not tested on GPU.

## Artifacts

- **Dataset collection:** https://huggingface.co/datasets/nirdesho6o/turndrift-control
- **Model collection (D2 vectors):** https://huggingface.co/nirdesho6o/turndrift-d2-vectors
- **Article:** https://huggingface.co/blog/nirdesho6o/turndrift-control
- **GitHub:** https://github.com/nirdesho6o/turndrift
- **Demo Space:** https://huggingface.co/spaces/nirdesho6o/turndrift-replay

## Kits used

1. **CuratorKIT** — Schema validation (SchemaGate), exact deduplication, embedding near-duplicate detection (DiversityGate), cross-split leak checks, provenance recording
2. **SafeTune** — D1 refusal direction extraction (extract_refusal_direction), D2 injection-contrast direction extraction (mean-difference CV), steering hook implementation
3. **AuditKIT** — Evaluation run recording with provenance chain (make_provenance, write_provenance), model fingerprints (sha256_digest), input references linking to CuratorKIT report

## Track and problem statement

**Track 2a** — Safety auditing and red-teaming

**Problem Statement 3** — Multi-turn audits: testing whether models follow hidden instructions embedded across multiple turns in agent conversations, and whether external monitors can detect and block injection compliance.

## Author

Nirdesh Bhandari (solo)
