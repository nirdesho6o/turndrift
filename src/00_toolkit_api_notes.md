# Toolkit API Notes (confirmed from docs, Nov 10 2026)

## CuratorKIT (v latest, Python 3.11+)

**Install:** `pip install "curatorkit[all]"`

**Imports:**
```python
from curatorkit import Curator, CuratorConfig
```

**Usage for validation:**
```python
config = CuratorConfig(
    dataset="path/to/data.jsonl",
    schema_gate=True,
    schema_required_fields=["episode_id", "turns", "injection_strategy"],
    dedup="exact",
    clean=True,
    output_dir="data/processed",
    export_formats=["sharegpt"],  # TODO: verify JSONL passthrough
)
result = Curator(config).run()
result.sample(3)
```

**Provenance:** Written automatically on every run — `manifest.json`, `lexsi_provenance.json`, `checksums.txt`, `rejected.jsonl`.

**Dataset card:** `DatasetCardGenerator` class exists but signature not documented. Card also auto-generated as `dataset_card.md`.

**Key config fields:** schema_gate, schema_required_fields, dedup, clean, output_dir, export_formats, output_split, hallucination_threshold, reward_threshold, diversity_threshold.

# TODO: verify CuratorConfig accepts raw list[dict] for dataset param
# TODO: verify export_formats supports plain jsonl passthrough


## SafeTune (v0.1.8, Python 3.12+)

**Install:** `pip install safetune`

**Imports:**
```python
from safetune.runner import steer
from safetune.steer import extract_refusal_direction, RefusalDirectionConfig, orthogonalize_weights, restore_weights
```

**Refusal direction extraction:**
```python
trainer = steer.RefusalDirectionTrainer(model, tokenizer, alpha=0.3)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)
```

Or lower-level:
```python
direction, _, _ = extract_refusal_direction(
    model, tokenizer, harmful_prompts, harmless_prompts,
    RefusalDirectionConfig(select_directions=False)
)
```

**Steering (context manager):**
```python
with wrapped:
    output = model.generate(...)
```

**Permanent weight edit:**
```python
snapshots = orthogonalize_weights(model, direction)
# restore later:
restore_weights(model, snapshots)
```

**Notes:**
- alpha default is 20 for RefusalDirectionTrainer; 0.3 worked for 0.5B model
- RefusalDirectionConfig(pick_layer=N) for manual layer selection
- **Custom contrast sets (confirmed Nov 10):** harmful/harmless args are plain lists of strings with no format constraint. D2 (injection vs clean) can use the same API by passing injection-bearing prefixes as "harmful" and clean twins as "harmless". HOWEVER: the validation sweep (`select_directions=True`, default) scores bypass/KL on refusal-style behavior, which won't fit injection-vs-clean contrast. For D2, use `RefusalDirectionConfig(select_directions=False)` or `pick_layer=N` to skip the refusal-specific sweep.
- RefusalDirectionConfig fields: target_layers, pick_layer, pool_method ("last_token"|"mean"), strength (1.0), normalize (True), select_directions (True), kl_threshold (0.1), n_val (16), max_new_tokens (24).
- RefusalDirectionTrainer(model, tokenizer, *, layers=None, alpha=20.0, orthogonalize=False)
# TODO: verify alpha vs strength naming (docs are inconsistent)


## AuditKIT (v1.1.2 per README, v1.0.0 per docs)

**Install:** `pip install auditkit`

**Imports:**
```python
import auditkit as ak
```

**Core evaluate:**
```python
result = ak.evaluate(
    dataset=samples,           # list[ak.Sample] or Scenario
    model=model_or_callable,   # Model | Callable | str like "hf:model_id"
    scorers=[...],             # metric names, Metric objects, or callables
    experiment_name="turndrift_base",
    tags={"condition": "base"},
)
```

**Custom scorer:**
```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)
def injection_compliance(sample, output):
    # return float or ak.Score
    ...
```

**Sample:**
```python
ak.Sample(input="...", target="...", metadata={...})
```

**RunResult methods:**
- `result.summary()` -> str
- `result.save(path, fmt='json')` -> None
- `result.push_to_hub(repo_id)` -> str
- `result.metric_table()` -> list[dict]

**Comparison:**
```python
comparison = ak.compare(baseline_result, candidate_result)
comparison.grades()  # dict[str, DeltaGrade]
comparison.grade()   # worst grade = ship/no-ship
```

**Model wrapping:**
```python
model = ak.CallableModel(fn=my_generate_fn, name="tiny-aya-base")
```

# TODO: verify ak.Direction enum exists
# TODO: verify Sample constructor fields
