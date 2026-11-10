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


def _generate(messages, max_new_tokens=512):
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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


demo = gr.Interface(
    fn=run_episode,
    inputs=gr.Textbox(label="Episode JSON", lines=10),
    outputs=gr.Textbox(label="Transcript JSON", lines=10),
    title="Aya Expanse 8B — TurnDrift Episode Runner",
    description="Paste one episode JSON. Returns full transcript with compliance detection.",
    api_name="run_episode",
)

if __name__ == "__main__":
    demo.launch()
