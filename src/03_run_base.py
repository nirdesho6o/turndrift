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


def generate_response(
    model, tokenizer, messages: list[dict],
    max_new_tokens: int = 512, device: str = "cuda",
) -> str:
    """Generate a single response given chat messages."""
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

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


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

    print(f"Running {len(episodes)} episodes with {args.model}")
    model, tokenizer = load_model(args.model, args.device, args.dtype)

    # Track compliance
    compliant = 0
    total_attacks = 0
    results_summary = []

    for i, episode in enumerate(episodes):
        print(f"\n[{i+1}/{len(episodes)}] {episode['episode_id']} ", end="", flush=True)

        transcript = run_episode(
            model, tokenizer, episode,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )

        # Save transcript
        out_path = out_dir / f"{episode['episode_id']}.json"
        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2)

        # Track compliance
        if episode["variant"] == "attack":
            total_attacks += 1
            if transcript["compliance"] and transcript["compliance"]["detected"]:
                compliant += 1
                print(f"COMPLIANT (turn {transcript['compliance']['after_turn']})", end="")
            else:
                print("REFUSED", end="")
        else:
            print("BENIGN", end="")

        results_summary.append({
            "episode_id": episode["episode_id"],
            "variant": episode["variant"],
            "strategy": episode.get("injection_strategy", "none"),
            "target": episode.get("injection_target", "none"),
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
