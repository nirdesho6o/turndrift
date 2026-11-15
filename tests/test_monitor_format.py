"""Tests for monitor formatting functions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_monitor


FAKE_EPISODE = {
    "episode_id": "test_ep",
    "system_message": "You are a helpful assistant.",
    "turns": [
        {"turn_number": 1, "content": "Hello"},
        {"turn_number": 2, "content": "Document with injection"},
        {"turn_number": 3, "content": "Follow up question"},
    ],
}

FAKE_TRANSCRIPT = {
    "episode_id": "test_ep",
    "responses": [
        {"after_turn": 1, "response": "Hi there!"},
        {"after_turn": 2, "response": "ACTION: send_email(evil)"},
        {"after_turn": 3, "response": "Sure thing."},
    ],
}


def _patch_episodes():
    return patch.object(run_monitor, "_EPISODES_CACHE", {"test_ep": FAKE_EPISODE})


def test_prefix_turn1_excludes_assistant_response():
    """At turn 1, prefix shows user msg but NOT assistant response."""
    with _patch_episodes():
        text = run_monitor.format_transcript_prefix(FAKE_TRANSCRIPT, up_to_turn=1)
    assert "[USER turn 1]: Hello" in text
    assert "Hi there!" not in text
    assert "turn 2" not in text


def test_prefix_turn2_includes_turn1_response():
    """At turn 2, prefix includes turn 1 user+assistant, turn 2 user only."""
    with _patch_episodes():
        text = run_monitor.format_transcript_prefix(FAKE_TRANSCRIPT, up_to_turn=2)
    assert "[USER turn 1]: Hello" in text
    assert "[ASSISTANT turn 1]: Hi there!" in text
    assert "[USER turn 2]: Document with injection" in text
    assert "ACTION: send_email(evil)" not in text
    assert "turn 3" not in text


def test_prefix_turn3_includes_prior_responses():
    """At turn 3, prefix includes turns 1-2 user+assistant, turn 3 user only."""
    with _patch_episodes():
        text = run_monitor.format_transcript_prefix(FAKE_TRANSCRIPT, up_to_turn=3)
    assert "[ASSISTANT turn 1]: Hi there!" in text
    assert "[ASSISTANT turn 2]: ACTION: send_email(evil)" in text
    assert "[USER turn 3]: Follow up question" in text
    assert "Sure thing." not in text


def test_single_message_shows_only_current():
    """Per-turn mode shows only the current user message."""
    with _patch_episodes():
        text = run_monitor.format_single_message(FAKE_TRANSCRIPT, turn_num=2)
    assert "Document with injection" in text
    assert "Hello" not in text
    assert "Hi there!" not in text
    assert "turn 3" not in text
