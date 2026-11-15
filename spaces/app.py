"""ZeroGPU Space for aya-expanse-8b inference (nirdesho6o/aya8b-check).

Exposes a /run_episode endpoint that takes one episode JSON and returns
the full transcript with compliance detection. Designed for ZeroGPU with
@spaces.GPU(duration=...) sized for 6-turn episodes.

Deploy: copy this file + detectors.py to the HF Space.
"""

import json
import time

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "CohereLabs/aya-expanse-8b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Detectors are vendored into the Space
from detectors import detect


def _tokenize_messages(messages):
    try:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
        return inputs["input_ids"][0], inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0]
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
        return inputs["input_ids"][0], inputs["attention_mask"][0]


def _generate(messages, max_new_tokens=512):
    ids, mask = _tokenize_messages(messages)
    ids = ids.unsqueeze(0).to(model.device)
    mask = mask.unsqueeze(0).to(model.device)
    prompt_len = ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_batched(message_lists, max_new_tokens=512):
    if not message_lists:
        return []

    all_ids = []
    all_masks = []
    for msgs in message_lists:
        ids, mask = _tokenize_messages(msgs)
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
        new_tokens = outputs[i][max_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


@spaces.GPU(duration=180)
def run_episode(episode_json: str) -> str:
    """Run a single episode through the 8B model. Returns transcript JSON."""
    episode = json.loads(episode_json)

    responses = []
    messages_so_far = [{"role": "system", "content": episode["system_message"]}]

    for turn in episode["turns"]:
        messages_so_far.append({"role": "user", "content": turn["content"]})

        t0 = time.time()
        response = _generate(messages_so_far, max_new_tokens=512)
        elapsed = time.time() - t0

        responses.append({
            "after_turn": turn["turn_number"],
            "response": response,
            "generation_time_s": round(elapsed, 2),
        })
        messages_so_far.append({"role": "assistant", "content": response})

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

    transcript = {
        "episode_id": episode["episode_id"],
        "pair_id": episode["pair_id"],
        "variant": episode["variant"],
        "scenario_domain": episode["scenario_domain"],
        "injection_strategy": episode.get("injection_strategy", "none"),
        "injection_target": episode.get("injection_target", "none"),
        "split": episode["split"],
        "condition": "base_8b",
        "responses": responses,
        "compliance": compliance,
    }
    return json.dumps(transcript)


def _estimate_batch_duration(batch_size: int, n_turns: int = 6) -> int:
    """Estimated GPU seconds for a batch: ~8s/turn scaled by batch_size/16, × 1.3 margin."""
    secs = int(n_turns * 8 * max(batch_size, 1) / 16 * 1.3)
    return max(secs, 60)


def run_episodes_batch(episodes_json: str) -> str:
    """Run a batch of episodes through the 8B model. Returns JSON array of transcripts.

    GPU duration is sized dynamically from batch size (estimated seconds × 1.3).
    ZeroGPU releases the GPU when the function returns — you pay for actual usage.
    """
    episodes = json.loads(episodes_json)
    if not episodes:
        return "[]"

    duration = _estimate_batch_duration(len(episodes), len(episodes[0]["turns"]))

    @spaces.GPU(duration=duration)
    def _inner():
        n_turns = len(episodes[0]["turns"])
        batch_size = len(episodes)

        messages_per_ep = [
            [{"role": "system", "content": ep["system_message"]}] for ep in episodes
        ]
        responses_per_ep = [[] for _ in episodes]

        for turn_idx in range(n_turns):
            for j, ep in enumerate(episodes):
                messages_per_ep[j].append({
                    "role": "user",
                    "content": ep["turns"][turn_idx]["content"],
                })

            t0 = time.time()
            batch_responses = _generate_batched(messages_per_ep, max_new_tokens=512)
            elapsed = time.time() - t0

            turn_num = episodes[0]["turns"][turn_idx]["turn_number"]
            for j in range(batch_size):
                responses_per_ep[j].append({
                    "after_turn": turn_num,
                    "response": batch_responses[j],
                    "generation_time_s": round(elapsed / batch_size, 2),
                })
                messages_per_ep[j].append({
                    "role": "assistant", "content": batch_responses[j],
                })

        transcripts = []
        for j, ep in enumerate(episodes):
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

            transcripts.append({
                "episode_id": ep["episode_id"],
                "pair_id": ep["pair_id"],
                "variant": ep["variant"],
                "scenario_domain": ep["scenario_domain"],
                "injection_strategy": ep.get("injection_strategy", "none"),
                "injection_target": ep.get("injection_target", "none"),
                "split": ep["split"],
                "condition": "base_8b",
                "responses": responses_per_ep[j],
                "compliance": compliance,
            })

        return transcripts

    transcripts = _inner()
    return json.dumps(transcripts)


with gr.Blocks(title="Aya Expanse 8B — TurnDrift Episode Runner") as demo:
    gr.Markdown("# Aya Expanse 8B — TurnDrift Episode Runner")
    gr.Markdown("Single episode or batch endpoint for TurnDrift-Control.")

    with gr.Tab("Single"):
        single_in = gr.Textbox(label="Episode JSON", lines=10)
        single_out = gr.Textbox(label="Transcript JSON", lines=10)
        single_btn = gr.Button("Run")
        single_btn.click(run_episode, inputs=single_in, outputs=single_out, api_name="run_episode")

    with gr.Tab("Batch"):
        batch_in = gr.Textbox(label="Episodes JSON (array)", lines=10)
        batch_out = gr.Textbox(label="Transcripts JSON (array)", lines=10)
        batch_btn = gr.Button("Run Batch")
        batch_btn.click(run_episodes_batch, inputs=batch_in, outputs=batch_out, api_name="run_episodes_batch")

if __name__ == "__main__":
    demo.launch()
