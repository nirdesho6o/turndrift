"""Dataset validator for TurnDrift-Control benchmark.

Checks:
1. Every pair has attack + benign twin
2. fragment_turns match contains_injection_fragment flags
3. No actual dangerous content (only proxy targets)
4. Dev/test split integrity
5. No test-domain content in calibration files
6. Summary statistics
"""

import json
import sys
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("data/processed/turndrift_core.jsonl")
SCHEMA_PATH = Path("data/schema.json")
CAL_DIR = Path("data/calibration")
TEST_DOMAINS = {"api_response_handler", "database_query", "file_processing"}


def load_episodes() -> list[dict]:
    episodes = []
    with open(DATASET_PATH) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def check_pairs(episodes: list[dict]) -> list[str]:
    errors = []
    pairs: dict[str, dict[str, dict]] = {}
    for ep in episodes:
        pid = ep["pair_id"]
        variant = ep["variant"]
        if pid not in pairs:
            pairs[pid] = {}
        if variant in pairs[pid]:
            errors.append(f"Duplicate {variant} for pair {pid}")
        pairs[pid][variant] = ep

    for pid, variants in pairs.items():
        if "attack" not in variants:
            errors.append(f"Pair {pid} missing attack episode")
        if "benign" not in variants:
            errors.append(f"Pair {pid} missing benign episode")
        if "attack" in variants and "benign" in variants:
            atk = variants["attack"]
            ben = variants["benign"]
            if atk["scenario_domain"] != ben["scenario_domain"]:
                errors.append(f"Pair {pid}: domain mismatch")
            if atk["split"] != ben["split"]:
                errors.append(f"Pair {pid}: split mismatch")
    return errors


def check_fragment_consistency(episodes: list[dict]) -> list[str]:
    errors = []
    for ep in episodes:
        frag_turns_meta = set(ep["fragment_turns"])
        frag_turns_actual = set()
        for turn in ep["turns"]:
            if turn["contains_injection_fragment"]:
                frag_turns_actual.add(turn["turn_number"])

        if ep["variant"] == "attack":
            if frag_turns_meta != frag_turns_actual:
                errors.append(
                    f"{ep['episode_id']}: fragment_turns metadata {frag_turns_meta} "
                    f"!= actual injection turns {frag_turns_actual}"
                )
            for turn in ep["turns"]:
                if turn["contains_injection_fragment"] and turn["fragment_content"] is None:
                    errors.append(
                        f"{ep['episode_id']} turn {turn['turn_number']}: "
                        f"injection flag set but fragment_content is null"
                    )
        elif ep["variant"] == "benign":
            if frag_turns_actual:
                errors.append(
                    f"{ep['episode_id']}: benign episode has injection fragments "
                    f"at turns {frag_turns_actual}"
                )
            if ep["injection_strategy"] != "none":
                errors.append(
                    f"{ep['episode_id']}: benign but strategy is {ep['injection_strategy']}"
                )
    return errors


def check_no_dangerous_content(episodes: list[dict]) -> list[str]:
    """Check that episodes only contain proxy targets, not real harmful content."""
    errors = []
    dangerous_patterns = [
        "how to make a bomb", "how to kill", "how to hack",
        "credit card number", "social security", "password is",
    ]
    for ep in episodes:
        for turn in ep["turns"]:
            content_lower = turn["content"].lower()
            for pattern in dangerous_patterns:
                if pattern in content_lower:
                    errors.append(
                        f"{ep['episode_id']} turn {turn['turn_number']}: "
                        f"potentially dangerous content: '{pattern}'"
                    )
    return errors


def check_split_integrity(episodes: list[dict]) -> list[str]:
    errors = []
    for ep in episodes:
        domain = ep["scenario_domain"]
        expected_split = "test" if domain in TEST_DOMAINS else "dev"
        if ep["split"] != expected_split:
            errors.append(
                f"{ep['episode_id']}: domain {domain} should be {expected_split} "
                f"but is {ep['split']}"
            )
    return errors


def check_calibration_no_test_leak() -> list[str]:
    """Verify no test-domain content in calibration files."""
    errors = []
    for cal_file in ["injection.jsonl", "clean.jsonl"]:
        cal_path = CAL_DIR / cal_file
        if not cal_path.exists():
            errors.append(f"Calibration file missing: {cal_path}")
            continue
        with open(cal_path) as f:
            for i, line in enumerate(f):
                entry = json.loads(line.strip())
                eid = entry.get("episode_id", "")
                for td in TEST_DOMAINS:
                    if td in eid:
                        errors.append(
                            f"{cal_file} line {i}: test domain '{td}' in episode {eid}"
                        )
    return errors


def check_turn_structure(episodes: list[dict]) -> list[str]:
    errors = []
    for ep in episodes:
        if len(ep["turns"]) != 6:
            errors.append(f"{ep['episode_id']}: expected 6 turns, got {len(ep['turns'])}")
            continue
        for i, turn in enumerate(ep["turns"]):
            expected_num = i + 1
            if turn["turn_number"] != expected_num:
                errors.append(
                    f"{ep['episode_id']} turn {i}: number is {turn['turn_number']}, "
                    f"expected {expected_num}"
                )
            expected_speaker = "user" if expected_num in (1, 3, 6) else "tool_output"
            if turn["speaker"] != expected_speaker:
                errors.append(
                    f"{ep['episode_id']} turn {expected_num}: speaker is "
                    f"'{turn['speaker']}', expected '{expected_speaker}'"
                )
    return errors


def print_summary(episodes: list[dict]) -> None:
    attacks = [e for e in episodes if e["variant"] == "attack"]
    benign = [e for e in episodes if e["variant"] == "benign"]

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episodes)} ({len(attacks)} attack, {len(benign)} benign)")
    print(f"Total pairs: {len(attacks)}")

    print(f"\nBy strategy:")
    for strat, count in sorted(Counter(e["injection_strategy"] for e in attacks).items()):
        print(f"  {strat}: {count}")

    print(f"\nBy target:")
    for tgt, count in sorted(Counter(e["injection_target"] for e in attacks).items()):
        print(f"  {tgt}: {count}")

    print(f"\nBy domain:")
    for domain, count in sorted(Counter(e["scenario_domain"] for e in attacks).items()):
        split = "test" if domain in TEST_DOMAINS else "dev"
        print(f"  {domain} ({split}): {count}")

    dev_atk = sum(1 for e in attacks if e["split"] == "dev")
    test_atk = sum(1 for e in attacks if e["split"] == "test")
    print(f"\nSplit: {dev_atk} dev attack, {test_atk} test attack")

    # Calibration summary
    for fname in ["harmful.jsonl", "harmless.jsonl", "injection.jsonl", "clean.jsonl"]:
        fpath = CAL_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                n = sum(1 for _ in f)
            print(f"  calibration/{fname}: {n} entries")


def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Run src/benchmark_generator.py first.")
        sys.exit(1)

    episodes = load_episodes()
    all_errors = []

    checks = [
        ("Pair completeness", check_pairs),
        ("Fragment consistency", check_fragment_consistency),
        ("No dangerous content", check_no_dangerous_content),
        ("Split integrity", check_split_integrity),
        ("Turn structure", check_turn_structure),
    ]

    for name, check_fn in checks:
        errors = check_fn(episodes)
        if errors:
            print(f"\n[FAIL] {name}: {len(errors)} errors")
            for e in errors[:5]:
                print(f"  - {e}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
        else:
            print(f"[PASS] {name}")
        all_errors.extend(errors)

    # Calibration check
    cal_errors = check_calibration_no_test_leak()
    if cal_errors:
        print(f"\n[FAIL] Calibration no-test-leak: {len(cal_errors)} errors")
        for e in cal_errors[:5]:
            print(f"  - {e}")
    else:
        print(f"[PASS] Calibration no-test-leak")
    all_errors.extend(cal_errors)

    print_summary(episodes)

    if all_errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED: {len(all_errors)} total errors")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print(f"VALIDATION PASSED: all checks green")


if __name__ == "__main__":
    main()
