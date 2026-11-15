"""Base model runner for TurnDrift-Control.

Replays all benchmark episodes through the target model with greedy decoding.
Saves transcripts and runs deterministic compliance detectors.

Usage (Colab T4):
    python src/03_run_base.py --model CohereLabs/tiny-aya-global --device cuda
    python src/03_run_base.py --model CohereLabs/tiny-aya-global --device cpu --limit 3

Usage (local CPU, quick test):
    python src/03_run_base.py --device cpu --limit 3
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from detectors import detect, CANARY_TOKEN


def load_episodes(path: str = "data/processed/turndrift_core.jsonl") -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def load_model(model_id: str, device: str = "cuda", dtype: str = "float16"):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "4bit": "4bit",
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    print(f"Loading {model_id} on {device} with {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if dtype == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to("cpu")

    model.eval()
    print(f"Model loaded: {model.config._name_or_path}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model, tokenizer


def build_chat_messages(episode: dict, up_to_turn: int) -> list[dict]:
    """Build chat messages list from episode turns up to a given turn number."""
    messages = [{"role": "system", "content": episode["system_message"]}]

    for turn in episode["turns"]:
        if turn["turn_number"] > up_to_turn:
            break
        if turn["speaker"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["speaker"] == "tool_output":
            messages.append({"role": "user", "content": turn["content"]})

    return messages


def _tokenize_messages(tokenizer, messages):
    """Tokenize a single message list, returning input_ids and attention_mask tensors."""
    import torch
    try:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    return inputs["input_ids"][0], inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0]


def generate_response(
    model, tokenizer, messages: list[dict],
    max_new_tokens: int = 512, device: str = "cuda",
) -> str:
    """Generate a single response given chat messages."""
    import torch

    input_ids, attn_mask = _tokenize_messages(tokenizer, messages)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attn_mask = attn_mask.unsqueeze(0).to(model.device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def generate_response_batched(
    model, tokenizer, message_lists: list[list[dict]],
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate responses for a batch of message lists. Left-pads for greedy decoding."""
    import torch

    if not message_lists:
        return []

    all_ids = []
    all_masks = []
    for msgs in message_lists:
        ids, mask = _tokenize_messages(tokenizer, msgs)
        all_ids.append(ids)
        all_masks.append(mask)

    max_len = max(ids.shape[0] for ids in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_ids = []
    padded_masks = []
    prompt_lens = []
    for ids, mask in zip(all_ids, all_masks):
        seq_len = ids.shape[0]
        prompt_lens.append(seq_len)
        pad_len = max_len - seq_len
        if pad_len > 0:
            padded_ids.append(torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids]))
            padded_masks.append(torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask]))
        else:
            padded_ids.append(ids)
            padded_masks.append(mask)

    batch_ids = torch.stack(padded_ids).to(model.device)
    batch_masks = torch.stack(padded_masks).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_ids,
            attention_mask=batch_masks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=pad_id,
        )

    responses = []
    for i, plen in enumerate(prompt_lens):
        pad_len = max_len - plen
        new_tokens = outputs[i][max_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return responses


def run_episodes_batched(
    model, tokenizer, episodes: list[dict],
    max_new_tokens: int = 512, batch_size: int = 16,
    device: str = "cuda", condition: str = "base",
) -> list[dict]:
    """Run multiple episodes in lock-step batches, turn by turn.

    All episodes must have the same number of turns. Returns list of transcripts.
    Auto-reduces batch_size on OOM.
    """
    import torch
    from detectors import detect

    if not episodes:
        return []

    n_turns = len(episodes[0]["turns"])

    # effective_bs persists across chunks and turns so OOM reductions stick
    effective_bs = batch_size

    all_transcripts = []
    for chunk_start in range(0, len(episodes), batch_size):
        chunk = episodes[chunk_start:chunk_start + batch_size]
        bs = len(chunk)

        messages_per_ep = [
            [{"role": "system", "content": ep["system_message"]}] for ep in chunk
        ]
        responses_per_ep = [[] for _ in chunk]

        for turn_idx in range(n_turns):
            for j, ep in enumerate(chunk):
                messages_per_ep[j].append({
                    "role": "user",
                    "content": ep["turns"][turn_idx]["content"],
                })

            t0 = time.time()
            current_bs = min(effective_bs, bs)
            batch_responses = None
            while current_bs >= 1:
                try:
                    batch_responses = []
                    for sub_start in range(0, bs, current_bs):
                        sub_msgs = messages_per_ep[sub_start:sub_start + current_bs]
                        sub_resp = generate_response_batched(
                            model, tokenizer, sub_msgs, max_new_tokens=max_new_tokens,
                        )
                        batch_responses.extend(sub_resp)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)
                    effective_bs = current_bs
                    print(f"\n  OOM: reducing effective batch size to {effective_bs}")
                    if current_bs == 1:
                        batch_responses = []
                        for j in range(bs):
                            resp = generate_response(
                                model, tokenizer, messages_per_ep[j],
                                max_new_tokens=max_new_tokens, device=device,
                            )
                            batch_responses.append(resp)
                        break

            elapsed = time.time() - t0
            turn_num = chunk[0]["turns"][turn_idx]["turn_number"]

            for j in range(bs):
                responses_per_ep[j].append({
                    "after_turn": turn_num,
                    "response": batch_responses[j],
                    "generation_time_s": round(elapsed / bs, 2),
                })
                messages_per_ep[j].append({
                    "role": "assistant", "content": batch_responses[j],
                })

        for j, ep in enumerate(chunk):
            compliance = None
            if ep["variant"] == "attack":
                target = ep["injection_target"]
                for resp in responses_per_ep[j]:
                    detected, evidence = detect(target, resp["response"])
                    if detected:
                        compliance = {
                            "detected": True,
                            "after_turn": resp["after_turn"],
                            "evidence": evidence,
                            "target": target,
                        }
                        break
                if compliance is None:
                    compliance = {"detected": False, "target": target}

            all_transcripts.append({
                "episode_id": ep["episode_id"],
                "pair_id": ep["pair_id"],
                "variant": ep["variant"],
                "scenario_domain": ep["scenario_domain"],
                "injection_strategy": ep.get("injection_strategy", "none"),
                "injection_target": ep.get("injection_target", "none"),
                "split": ep["split"],
                "condition": condition,
                "responses": responses_per_ep[j],
                "compliance": compliance,
                BATCH_CONFIG_KEY: _make_batch_config(batch_size, True),
            })

    return all_transcripts


def verify_batched_outputs(model, tokenizer, episodes, device="cuda", max_new_tokens=512,
                           results_dir=None):
    """Verify batched greedy outputs match unbatched on up to 3 episodes.

    Logs results to results_dir/batch_equivalence.json. Returns (match, report_dict).
    Does NOT auto-fall back — caller decides whether to continue batched.
    """
    check_eps = episodes[:3]
    if not check_eps:
        report = {"match": True, "checked": 0, "message": "no episodes to check"}
        return True, report

    unbatched = []
    for ep in check_eps:
        t = run_episode(model, tokenizer, ep, max_new_tokens=max_new_tokens, device=device)
        unbatched.append(t)

    batched = run_episodes_batched(
        model, tokenizer, check_eps,
        max_new_tokens=max_new_tokens, batch_size=len(check_eps), device=device,
    )

    mismatches = []
    for i, (ub, bt) in enumerate(zip(unbatched, batched)):
        for turn_idx, (ur, br) in enumerate(zip(ub["responses"], bt["responses"])):
            if ur["response"] != br["response"]:
                ub_ids = tokenizer.encode(ur["response"])
                bt_ids = tokenizer.encode(br["response"])
                first_diff = 0
                for k in range(min(len(ub_ids), len(bt_ids))):
                    if ub_ids[k] != bt_ids[k]:
                        first_diff = k
                        break
                else:
                    first_diff = min(len(ub_ids), len(bt_ids))

                mismatches.append({
                    "episode": check_eps[i]["episode_id"],
                    "turn": ur["after_turn"],
                    "first_divergent_token_pos": first_diff,
                    "unbatched_prefix": ur["response"][:200],
                    "batched_prefix": br["response"][:200],
                    "unbatched_len_tokens": len(ub_ids),
                    "batched_len_tokens": len(bt_ids),
                })
                break

    mismatch_rate = len(mismatches) / len(check_eps) if check_eps else 0
    report = {
        "match": len(mismatches) == 0,
        "checked": len(check_eps),
        "mismatch_rate": mismatch_rate,
        "mismatches": mismatches,
    }

    if results_dir is not None:
        from pathlib import Path
        out = Path(results_dir) / "batch_equivalence.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Batch equivalence log: {out}")

    return len(mismatches) == 0, report


# Batch config recorded in every transcript for consistency enforcement
BATCH_CONFIG_KEY = "_batch_config"

def _make_batch_config(batch_size: int, batched: bool) -> dict:
    return {"batch_size": batch_size, "padding": "left" if batched else "none", "batched": batched}

def check_batch_config_consistency(results_dir, expected_config: dict):
    """Check that no existing transcripts in results_dir use a different batch config.

    Raises RuntimeError if a mismatch is found.
    """
    from pathlib import Path
    rd = Path(results_dir)
    for cond_dir in (rd / "transcripts").iterdir() if (rd / "transcripts").exists() else []:
        if not cond_dir.is_dir():
            continue
        for p in cond_dir.glob("*.json"):
            if p.stem == "_summary":
                continue
            with open(p) as f:
                t = json.load(f)
            existing = t.get(BATCH_CONFIG_KEY)
            if existing is not None and existing != expected_config:
                raise RuntimeError(
                    f"Batch config mismatch in {p.relative_to(rd)}: "
                    f"existing={existing}, expected={expected_config}. "
                    f"All conditions must use the same batch size and padding config. "
                    f"Delete existing transcripts or use the same --batch-size."
                )


def run_episode(
    model, tokenizer, episode: dict,
    max_new_tokens: int = 512, device: str = "cuda",
) -> dict:
    """Run a single episode and collect all model responses."""
    responses = []
    messages_so_far = [{"role": "system", "content": episode["system_message"]}]

    # User/tool turns are 1, 2, 3, 4, 5, 6
    # Model responds after each turn
    for turn in episode["turns"]:
        messages_so_far.append({
            "role": "user",
            "content": turn["content"],
        })

        t0 = time.time()
        response = generate_response(
            model, tokenizer, messages_so_far,
            max_new_tokens=max_new_tokens, device=device,
        )
        elapsed = time.time() - t0

        responses.append({
            "after_turn": turn["turn_number"],
            "response": response,
            "generation_time_s": round(elapsed, 2),
        })

        messages_so_far.append({"role": "assistant", "content": response})

    # Run compliance detection on attack episodes
    compliance = None
    if episode["variant"] == "attack":
        target = episode["injection_target"]
        for resp in responses:
            detected, evidence = detect(target, resp["response"])
            if detected:
                compliance = {
                    "detected": True,
                    "after_turn": resp["after_turn"],
                    "evidence": evidence,
                    "target": target,
                }
                break
        if compliance is None:
            compliance = {"detected": False, "target": target}

    return {
        "episode_id": episode["episode_id"],
        "pair_id": episode["pair_id"],
        "variant": episode["variant"],
        "scenario_domain": episode["scenario_domain"],
        "injection_strategy": episode.get("injection_strategy", "none"),
        "injection_target": episode.get("injection_target", "none"),
        "split": episode["split"],
        "condition": "base",
        "responses": responses,
        "compliance": compliance,
        BATCH_CONFIG_KEY: _make_batch_config(1, False),
    }


def main():
    parser = argparse.ArgumentParser(description="Run base model on TurnDrift-Control")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dataset", default="data/processed/turndrift_core.jsonl")
    parser.add_argument("--output-dir", default="results/transcripts/base")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    parser.add_argument("--split", default=None, choices=["dev", "test"],
                        help="Only run episodes from this split")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip episodes with existing transcripts")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for parallel generation (default 4 for T4 15GB)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.dataset)
    if args.split:
        episodes = [e for e in episodes if e["split"] == args.split]
    if args.limit:
        episodes = episodes[:args.limit]

    # Skip existing
    if args.skip_existing:
        existing = {p.stem for p in out_dir.glob("*.json")}
        before = len(episodes)
        episodes = [e for e in episodes if e["episode_id"] not in existing]
        if before != len(episodes):
            print(f"Skipping {before - len(episodes)} existing transcripts")

    if not episodes:
        print("No episodes to run.")
        return

    print(f"Running {len(episodes)} episodes with {args.model} (batch_size={args.batch_size})")
    model, tokenizer = load_model(args.model, args.device, args.dtype)

    # Verify batched == unbatched on 3 episodes (log, don't fall back)
    if args.batch_size > 1 and len(episodes) >= 1:
        print("\nVerifying batched vs unbatched outputs on up to 3 episodes...")
        match, report = verify_batched_outputs(
            model, tokenizer, episodes, device=args.device,
            max_new_tokens=args.max_new_tokens,
            results_dir=str(Path(args.output_dir).parent.parent),
        )
        if match:
            print(f"  PASS: {report['checked']} episodes match")
        else:
            print(f"  MISMATCH ({report['mismatch_rate']:.0%} of checked episodes):")
            for m in report["mismatches"]:
                print(f"    {m['episode']} turn {m['turn']}: "
                      f"first divergent token at pos {m['first_divergent_token_pos']}")
            print("  Continuing batched — mismatch logged to batch_equivalence.json")

    n_turns = len(episodes[0]["turns"]) if episodes else 6
    est_gens = len(episodes) * n_turns
    est_mins = est_gens * 5 / 60  # ~5s/gen on T4 with batching
    print(f"\nEstimated: {est_gens} generations (~{est_mins:.0f} min on T4)")

    # Track compliance
    compliant = 0
    total_attacks = 0
    results_summary = []

    for chunk_start in range(0, len(episodes), args.batch_size):
        chunk = episodes[chunk_start:chunk_start + args.batch_size]
        print(f"\nBatch {chunk_start // args.batch_size + 1}: "
              f"episodes {chunk_start + 1}-{chunk_start + len(chunk)} of {len(episodes)}")

        if args.batch_size > 1:
            transcripts = run_episodes_batched(
                model, tokenizer, chunk,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                device=args.device,
            )
        else:
            transcripts = [
                run_episode(model, tokenizer, ep,
                            max_new_tokens=args.max_new_tokens, device=args.device)
                for ep in chunk
            ]

        for transcript in transcripts:
            eid = transcript["episode_id"]
            out_path = out_dir / f"{eid}.json"
            with open(out_path, "w") as f:
                json.dump(transcript, f, indent=2)

            ep = next(e for e in chunk if e["episode_id"] == eid)
            if ep["variant"] == "attack":
                total_attacks += 1
                if transcript["compliance"] and transcript["compliance"]["detected"]:
                    compliant += 1
                    print(f"  {eid}: COMPLIANT (turn {transcript['compliance']['after_turn']})")
                else:
                    print(f"  {eid}: REFUSED")
            else:
                print(f"  {eid}: BENIGN")

            results_summary.append({
                "episode_id": eid,
                "variant": ep["variant"],
                "strategy": ep.get("injection_strategy", "none"),
                "target": ep.get("injection_target", "none"),
                "compliant": (transcript["compliance"]["detected"]
                              if transcript["compliance"] else None),
            })

    # Print summary
    print(f"\n\n{'='*60}")
    print(f"BASE MODEL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Episodes run: {len(results_summary)}")
    if total_attacks > 0:
        print(f"Attack episodes: {total_attacks}")
        print(f"Injection Compliance Rate: {compliant}/{total_attacks} "
              f"({100*compliant/total_attacks:.1f}%)")

        # By strategy
        print(f"\nBy strategy:")
        strategies = set(r["strategy"] for r in results_summary if r["variant"] == "attack")
        for strat in sorted(strategies):
            strat_results = [r for r in results_summary
                             if r["strategy"] == strat and r["variant"] == "attack"]
            strat_compliant = sum(1 for r in strat_results if r["compliant"])
            print(f"  {strat}: {strat_compliant}/{len(strat_results)} "
                  f"({100*strat_compliant/len(strat_results):.0f}%)")

        # By target
        print(f"\nBy target:")
        targets = set(r["target"] for r in results_summary if r["variant"] == "attack")
        for tgt in sorted(targets):
            tgt_results = [r for r in results_summary
                           if r["target"] == tgt and r["variant"] == "attack"]
            tgt_compliant = sum(1 for r in tgt_results if r["compliant"])
            print(f"  {tgt}: {tgt_compliant}/{len(tgt_results)} "
                  f"({100*tgt_compliant/len(tgt_results):.0f}%)")

    # Save summary
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "total_episodes": len(results_summary),
            "total_attacks": total_attacks,
            "compliant": compliant,
            "icr": compliant / total_attacks if total_attacks > 0 else 0,
            "results": results_summary,
        }, f, indent=2)

    print(f"\nTranscripts saved to {out_dir}/")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
