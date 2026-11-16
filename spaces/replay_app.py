"""CPU-only transcript replay viewer for TurnDrift-Control.

Loads precomputed transcripts and lets users browse episodes,
see turn-by-turn conversations, and view compliance/monitor results.

Deploy to HF Spaces (CPU, no GPU needed).
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


def list_episodes() -> list[str]:
    return sorted(EPISODES.keys())


def format_turn(turn: dict) -> str:
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    icon = {"system": "gear", "user": "person", "assistant": "robot"}.get(role, "chat")
    return f"**[{role.upper()}]**\n\n{content}"


def render_episode(episode_id: str) -> tuple[str, str, str]:
    if not episode_id or episode_id not in EPISODES:
        return "Select an episode.", "", ""

    ep = EPISODES[episode_id]
    meta = ep.get("metadata", {})

    header_parts = [f"**Episode:** `{episode_id}`"]
    if "strategy" in meta:
        header_parts.append(f"**Strategy:** {meta['strategy']}")
    if "target" in meta:
        header_parts.append(f"**Target:** {meta['target']}")
    if "domain" in meta:
        header_parts.append(f"**Domain:** {meta['domain']}")
    if "split" in meta:
        header_parts.append(f"**Split:** {meta['split']}")
    header = " | ".join(header_parts)

    turns = ep.get("transcript", ep.get("turns", []))
    conversation = "\n\n---\n\n".join(format_turn(t) for t in turns)

    results_parts = []
    compliance = ep.get("compliance", {})
    if compliance:
        label = compliance.get("label", "unknown")
        emoji = {"compliant": "YES", "safe": "NO"}.get(label, label.upper())
        results_parts.append(f"**Injection compliance:** {emoji}")
        if "after_turn" in compliance:
            results_parts.append(f"**First compliance at turn:** {compliance['after_turn']}")
        if "detector" in compliance:
            results_parts.append(f"**Detector:** {compliance['detector']}")

    monitor = ep.get("monitor", {})
    if monitor:
        verdict = monitor.get("verdict", "unknown")
        results_parts.append(f"**Monitor verdict:** {verdict}")
        if "first_block_turn" in monitor:
            results_parts.append(f"**First block at turn:** {monitor['first_block_turn']}")

    results = "\n\n".join(results_parts) if results_parts else "No evaluation results available."

    return header, conversation, results


def build_app():
    with gr.Blocks(
        title="TurnDrift-Control Replay",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# TurnDrift-Control — Transcript Replay\n\n"
            "Browse multi-turn prompt injection episodes and their evaluation results.\n\n"
            f"**{len(EPISODES)} episodes loaded.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                episode_dd = gr.Dropdown(
                    choices=list_episodes(),
                    label="Select episode",
                    interactive=True,
                )
                header_md = gr.Markdown("")
                results_md = gr.Markdown("### Results\n\nSelect an episode to see results.")

            with gr.Column(scale=2):
                conversation_md = gr.Markdown(
                    "### Conversation\n\nSelect an episode to view the transcript.",
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
