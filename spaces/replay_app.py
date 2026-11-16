"""CPU-only transcript replay viewer for TurnDrift-Control.

Loads precomputed episodes and model responses, lets users browse
turn-by-turn conversations and view compliance/monitor results.

Deploy to HF Spaces (CPU, no GPU needed).

For local testing:
    python spaces/replay_app.py

For HF Spaces deployment, copy this file plus a transcripts/ directory
containing merged JSON files (episode + responses + monitor verdicts).
Use scripts/export_replay.py to generate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

TRANSCRIPT_DIR = Path("transcripts")


def load_transcripts() -> dict[str, dict]:
    episodes = {}
    if not TRANSCRIPT_DIR.exists():
        return episodes
    for f in sorted(TRANSCRIPT_DIR.glob("*.json")):
        with open(f) as fh:
            ep = json.load(fh)
        episodes[ep.get("episode_id", f.stem)] = ep
    return episodes


EPISODES = load_transcripts()

STRATEGY_LABELS = {
    "single_turn": "Single turn",
    "two_fragment": "Two fragment",
    "three_fragment": "Three fragment",
    "semantic_split": "Semantic split",
    "delayed_trigger": "Delayed trigger",
}


def list_episodes(variant_filter: str = "all") -> list[str]:
    eps = sorted(EPISODES.keys())
    if variant_filter == "attack":
        eps = [e for e in eps if EPISODES[e].get("variant") == "attack"]
    elif variant_filter == "benign":
        eps = [e for e in eps if EPISODES[e].get("variant") == "benign"]
    return eps


def render_episode(episode_id: str) -> tuple[str, str, str]:
    if not episode_id or episode_id not in EPISODES:
        return "Select an episode from the dropdown.", "", ""

    ep = EPISODES[episode_id]

    # Header
    parts = [f"**Episode:** `{episode_id}`"]
    parts.append(f"**Variant:** {ep.get('variant', '?')}")
    if ep.get("injection_strategy"):
        parts.append(f"**Strategy:** {STRATEGY_LABELS.get(ep['injection_strategy'], ep['injection_strategy'])}")
    if ep.get("injection_target"):
        parts.append(f"**Target:** {ep['injection_target']}")
    parts.append(f"**Domain:** {ep.get('scenario_domain', '?')}")
    parts.append(f"**Split:** {ep.get('split', '?')}")
    header = "\n\n".join(parts)

    # Conversation
    turns = ep.get("turns", [])
    responses = {r["after_turn"]: r["response"] for r in ep.get("responses", [])}
    blocks = []

    for turn in turns:
        tn = turn["turn_number"]
        speaker = turn.get("speaker", "user")
        content = turn.get("content", "")

        has_frag = turn.get("contains_injection_fragment", False)
        frag_marker = " **[INJECTION FRAGMENT]**" if has_frag else ""

        blocks.append(f"### Turn {tn} — {speaker.upper()}{frag_marker}\n\n{content}")

        if tn in responses:
            blocks.append(f"### Turn {tn} — ASSISTANT (model response)\n\n{responses[tn]}")

    conversation = "\n\n---\n\n".join(blocks) if blocks else "No turns available."

    # Results
    result_parts = []
    compliance = ep.get("compliance", {})
    if compliance:
        detected = compliance.get("detected", False)
        result_parts.append(f"**Injection compliant:** {'YES' if detected else 'NO'}")
        if detected:
            result_parts.append(f"**First compliance at turn:** {compliance.get('after_turn', '?')}")
            if compliance.get("evidence"):
                result_parts.append(f"**Evidence:** {compliance['evidence']}")
        result_parts.append(f"**Target:** {compliance.get('target', '?')}")

    monitor = ep.get("monitor_full_context", {})
    if monitor:
        result_parts.append(f"\n**Full-context monitor verdict:** {monitor.get('verdict', '?')}")
        fb = monitor.get("first_block_turn")
        if fb is not None:
            result_parts.append(f"**First block at turn:** {fb}")

    monitor_pt = ep.get("monitor_per_turn", {})
    if monitor_pt:
        result_parts.append(f"\n**Per-turn monitor verdict:** {monitor_pt.get('verdict', '?')}")
        fb = monitor_pt.get("first_block_turn")
        if fb is not None:
            result_parts.append(f"**First block at turn:** {fb}")

    results = "\n\n".join(result_parts) if result_parts else "No evaluation results."

    return header, conversation, results


def filter_episodes(variant: str):
    return gr.update(choices=list_episodes(variant), value=None)


def build_app():
    with gr.Blocks(
        title="TurnDrift-Control Replay",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# TurnDrift-Control — Transcript Replay\n\n"
            "Browse multi-turn prompt injection episodes from the TurnDrift-Control benchmark.\n\n"
            f"**{len(EPISODES)} episodes loaded.**"
        )

        with gr.Row():
            variant_radio = gr.Radio(
                choices=["all", "attack", "benign"],
                value="all",
                label="Filter by variant",
            )

        with gr.Row():
            with gr.Column(scale=1):
                episode_dd = gr.Dropdown(
                    choices=list_episodes(),
                    label="Select episode",
                    interactive=True,
                )
                header_md = gr.Markdown("")
                results_md = gr.Markdown("### Results\n\nSelect an episode.")

            with gr.Column(scale=2):
                conversation_md = gr.Markdown(
                    "### Conversation\n\nSelect an episode to view the transcript.",
                )

        variant_radio.change(
            fn=filter_episodes,
            inputs=[variant_radio],
            outputs=[episode_dd],
        )

        episode_dd.change(
            fn=render_episode,
            inputs=[episode_dd],
            outputs=[header_md, conversation_md, results_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
