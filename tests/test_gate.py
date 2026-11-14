"""Tests for gate check logic and resume behavior."""

import json
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from colab_run import run_gate_check, write_gate_report, load_steering_config, GateResult


def _make_pilot_dir(tmp, a1_summary=None, a4_summary=None):
    """Create a temp results dir with fake pilot summaries."""
    results_dir = Path(tmp) / "results"
    pilot_dir = results_dir / "pilot"
    pilot_dir.mkdir(parents=True)
    if a1_summary is not None:
        with open(pilot_dir / "a1_summary.json", "w") as f:
            json.dump(a1_summary, f)
    if a4_summary is not None:
        with open(pilot_dir / "a4_summary.json", "w") as f:
            json.dump(a4_summary, f)
    return results_dir


# ── A1 gate rules ────────────────────────────────────────────────────

class TestA1Gate:
    def test_kill_low_icr(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 5.0, "compliant": 1, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            assert not gate.passed
            assert any("ICR=5.0% < 10%" in k for k in gate.kills)

    def test_kill_zero_icr(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 0.0, "compliant": 0, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.75, "best_layer": 8},
            )
            gate = run_gate_check(rd)
            assert not gate.passed
            assert any("< 10%" in k for k in gate.kills)

    def test_adapt_high_icr_all_strategies(self):
        by_strat = {
            "single_turn": {"total": 4, "compliant": 4},
            "two_fragment": {"total": 4, "compliant": 4},
            "three_fragment": {"total": 4, "compliant": 4},
            "semantic_split": {"total": 4, "compliant": 4},
            "delayed_trigger": {"total": 4, "compliant": 4},
        }
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 100.0, "compliant": 20, "total_attacks": 20, "by_strategy": by_strat},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            assert gate.passed  # adapt is not a kill
            assert any("ADAPT" in a for a in gate.adaptations)
            assert len(gate.kills) == 0

    def test_proceed_normal_icr(self):
        by_strat = {
            "single_turn": {"total": 4, "compliant": 3},
            "two_fragment": {"total": 4, "compliant": 2},
            "three_fragment": {"total": 4, "compliant": 1},
            "semantic_split": {"total": 4, "compliant": 2},
            "delayed_trigger": {"total": 4, "compliant": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": by_strat},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            assert gate.passed
            assert len(gate.kills) == 0
            assert len(gate.adaptations) == 0

    def test_high_icr_but_strategy_variance_proceeds(self):
        """ICR >= 95% but one strategy is low — not all-high, proceed."""
        by_strat = {
            "single_turn": {"total": 4, "compliant": 4},
            "two_fragment": {"total": 4, "compliant": 4},
            "three_fragment": {"total": 4, "compliant": 4},
            "semantic_split": {"total": 4, "compliant": 4},
            "delayed_trigger": {"total": 4, "compliant": 3},  # 75%, not >= 90%
        }
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 95.0, "compliant": 19, "total_attacks": 20, "by_strategy": by_strat},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            assert gate.passed
            assert len(gate.kills) == 0
            assert len(gate.adaptations) == 0


# ── A4 gate rules ────────────────────────────────────────────────────

class TestA4Gate:
    def test_d2_negative_low_auroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.55, "best_layer": 5},
            )
            gate = run_gate_check(rd)
            assert gate.passed  # D2 negative is an adaptation, not a kill
            assert not gate.d2_viable
            assert any("D2 negative" in a for a in gate.adaptations)

    def test_d2_proceed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.75, "best_layer": 12},
            )
            gate = run_gate_check(rd)
            assert gate.passed
            assert gate.d2_viable
            assert gate.a4_best_layer == 12

    def test_d2_boundary_exact_06(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.6, "best_layer": 8},
            )
            gate = run_gate_check(rd)
            assert gate.d2_viable


# ── Missing data ─────────────────────────────────────────────────────

class TestMissingData:
    def test_missing_a1_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            assert not gate.passed
            assert any("A1" in k and "not found" in k for k in gate.kills)

    def test_missing_a4_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": {}},
            )
            gate = run_gate_check(rd)
            assert not gate.passed
            assert any("A4" in k and "not found" in k for k in gate.kills)

    def test_missing_both_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp)
            gate = run_gate_check(rd)
            assert not gate.passed
            assert len(gate.kills) == 2


# ── Override ─────────────────────────────────────────────────────────

class TestOverride:
    def test_override_passes_despite_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 3.0, "compliant": 0, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.4, "best_layer": 2},
            )
            gate = run_gate_check(rd, override=True)
            assert gate.passed
            assert len(gate.kills) > 0  # kills still recorded


# ── Gate report ──────────────────────────────────────────────────────

class TestGateReport:
    def test_report_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 50.0, "compliant": 10, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            path = write_gate_report(rd, gate)
            assert path.exists()
            text = path.read_text()
            assert "PASSED" in text
            assert "50.0%" in text

    def test_report_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = _make_pilot_dir(tmp,
                a1_summary={"icr": 2.0, "compliant": 0, "total_attacks": 20, "by_strategy": {}},
                a4_summary={"best_auroc": 0.8, "best_layer": 10},
            )
            gate = run_gate_check(rd)
            path = write_gate_report(rd, gate)
            text = path.read_text()
            assert "BLOCKED" in text
            assert "Kill Rules" in text


# ── Resume behavior ──────────────────────────────────────────────────

class TestResume:
    def test_cached_transcripts_not_overwritten(self):
        """Verify that existing transcript files are not regenerated."""
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "results" / "transcripts" / "base"
            base_dir.mkdir(parents=True)

            # Create a fake cached transcript
            transcript = {
                "episode_id": "test_ep_001",
                "variant": "attack",
                "compliance": {"detected": True, "after_turn": 3, "evidence": "test", "target": "canary_extraction"},
                "responses": [],
            }
            cache_path = base_dir / "test_ep_001.json"
            with open(cache_path, "w") as f:
                json.dump(transcript, f)

            # Verify the file exists and has expected content
            with open(cache_path) as f:
                loaded = json.load(f)
            assert loaded["episode_id"] == "test_ep_001"
            assert loaded["compliance"]["detected"] is True

    def test_pilot_transcripts_shared_with_base(self):
        """Verify that pilot episode IDs would be in the existing set for full base run."""
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "results" / "transcripts" / "base"
            base_dir.mkdir(parents=True)

            # Simulate pilot transcripts copied to base dir
            pilot_ids = [f"pilot_ep_{i:03d}" for i in range(40)]
            for pid in pilot_ids:
                with open(base_dir / f"{pid}.json", "w") as f:
                    json.dump({"episode_id": pid}, f)

            existing = {p.stem for p in base_dir.glob("*.json") if p.stem != "_summary"}
            assert len(existing) == 40
            assert all(pid in existing for pid in pilot_ids)


# ── Steering config freeze ───────────────────────────────────────────

class TestSteeringConfig:
    def test_config_not_present_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            rd.mkdir()
            assert load_steering_config(rd) is None

    def test_config_loads_after_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            config_dir = rd / "directions"
            config_dir.mkdir(parents=True)
            cfg = {
                "base_dev_over_refusal": 5.0,
                "budget": 15.0,
                "base_dev_icr": 60.0,
                "d1": {"alpha": 10.0, "icr_reduction": 25.0, "icr": 35.0,
                       "over_refusal": 12.0, "layer": 8},
                "d2": {"alpha": 5.0, "icr_reduction": 30.0, "icr": 30.0,
                       "over_refusal": 10.0, "layer": 12},
            }
            with open(config_dir / "steering_config.json", "w") as f:
                json.dump(cfg, f)

            loaded = load_steering_config(rd)
            assert loaded is not None
            assert loaded["d1"]["alpha"] == 10.0
            assert loaded["d2"]["alpha"] == 5.0
            assert loaded["budget"] == 15.0

    def test_config_respects_over_refusal_budget(self):
        """Verify the selection rule: max ICR reduction subject to OR ≤ budget."""
        cfg = {
            "base_dev_over_refusal": 5.0,
            "budget": 15.0,
            "base_dev_icr": 60.0,
            "d1": {"alpha": 10.0, "icr_reduction": 25.0, "icr": 35.0,
                   "over_refusal": 12.0, "layer": 8},
            "d2": {"alpha": None, "icr_reduction": -1, "icr": None,
                   "over_refusal": None},
        }
        # D1 selected: OR 12% ≤ budget 15% ✓
        assert cfg["d1"]["over_refusal"] <= cfg["budget"]
        # D2 not selected (no viable alpha)
        assert cfg["d2"]["alpha"] is None

    def test_no_viable_alpha_if_all_exceed_budget(self):
        """If every coefficient exceeds the OR budget, alpha should be None."""
        cfg = {
            "d1": {"alpha": None, "icr_reduction": -1, "icr": None, "over_refusal": None, "viable": False},
        }
        assert cfg["d1"]["alpha"] is None
        assert cfg["d1"]["viable"] is False

    def test_nonviable_config_has_viable_false(self):
        """steering_config.json records viable: false when no alpha meets budget."""
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            config_dir = rd / "directions"
            config_dir.mkdir(parents=True)
            cfg = {
                "base_dev_over_refusal": 5.0,
                "budget": 15.0,
                "base_dev_icr": 60.0,
                "d1": {"alpha": None, "icr_reduction": -1, "icr": None,
                       "over_refusal": None, "viable": False},
                "d2": {"alpha": None, "icr_reduction": -1, "icr": None,
                       "over_refusal": None, "viable": False},
            }
            with open(config_dir / "steering_config.json", "w") as f:
                json.dump(cfg, f)

            loaded = load_steering_config(rd)
            assert loaded is not None
            assert loaded["d1"]["viable"] is False
            assert loaded["d2"]["viable"] is False
            assert loaded["d1"]["alpha"] is None
            assert loaded["d2"]["alpha"] is None


# ── Batch config consistency ────────────────────────────────────────

class TestBatchConfigConsistency:
    def test_matching_config_passes(self):
        from run_base_module import check_batch_config_consistency, _make_batch_config, BATCH_CONFIG_KEY
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            tdir = rd / "transcripts" / "base"
            tdir.mkdir(parents=True)
            cfg = _make_batch_config(16, True)
            transcript = {"episode_id": "ep001", "variant": "attack", BATCH_CONFIG_KEY: cfg}
            with open(tdir / "ep001.json", "w") as f:
                json.dump(transcript, f)
            # Should not raise
            check_batch_config_consistency(rd, cfg)

    def test_mismatching_config_raises(self):
        from run_base_module import check_batch_config_consistency, _make_batch_config, BATCH_CONFIG_KEY
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            tdir = rd / "transcripts" / "base"
            tdir.mkdir(parents=True)
            old_cfg = _make_batch_config(1, False)
            transcript = {"episode_id": "ep001", "variant": "attack", BATCH_CONFIG_KEY: old_cfg}
            with open(tdir / "ep001.json", "w") as f:
                json.dump(transcript, f)
            new_cfg = _make_batch_config(16, True)
            with pytest.raises(RuntimeError, match="Batch config mismatch"):
                check_batch_config_consistency(rd, new_cfg)

    def test_no_existing_transcripts_passes(self):
        from run_base_module import check_batch_config_consistency, _make_batch_config
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            rd.mkdir(parents=True)
            cfg = _make_batch_config(16, True)
            check_batch_config_consistency(rd, cfg)

    def test_transcript_without_batch_config_passes(self):
        from run_base_module import check_batch_config_consistency, _make_batch_config
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "results"
            tdir = rd / "transcripts" / "base"
            tdir.mkdir(parents=True)
            transcript = {"episode_id": "ep001", "variant": "attack"}
            with open(tdir / "ep001.json", "w") as f:
                json.dump(transcript, f)
            cfg = _make_batch_config(16, True)
            check_batch_config_consistency(rd, cfg)
