"""Tests for tools/shader_diff.py (no game data, no GPU, no llvm-mc).

Run from the repository root:
    python -m pytest tests/test_shader_diff.py -v
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_diff as sd


def test_program_generation_is_deterministic():
    import random
    rng1, rng2 = random.Random(99), random.Random(99)
    assert (sd.generate_program(rng1, 6)[0] ==
            sd.generate_program(rng2, 6)[0])


def test_compare_pair_flags_known_bad():
    bad = sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 5])
    assert bad is not None and "lane word 3" in bad
    assert sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 4]) is None


def test_compare_pair_both_nan_matches():
    qnan = 0x7FC00001
    assert sd.compare_pair([qnan] * 4, [0x7FC00002] * 4) is None
    bad = sd.compare_pair([qnan] * 4, [0x3F800000] * 4)
    assert bad is not None and "NaN" in bad


def test_unsupported_op_is_a_counted_skip():
    status, detail = sd.run_one(
        "ps_probe_skip", ["v_interp_p1_f32 v8, v9, v10"],
        {8: ("v_interp_p1_f32", 1)})
    assert status == "skip"
    assert detail


def test_small_fixed_seed_run_is_clean():
    counts, first_failure = sd.run_suite(5, seed=20261121, max_ops=4)
    assert counts["mismatch"] == 0, first_failure
    assert counts["harness-error"] == 0, first_failure
    assert counts["pass"] == 5, counts


def test_self_test_passes():
    assert sd.self_test() is True

@pytest.mark.parametrize(("option", "value"), [
    ("--programs", "0"), ("--programs", "-3"),
    ("--max-ops", "0"), ("--max-ops", "-2"),
    ("--limit", "0"), ("--limit", "-2"),
])
def test_cli_rejects_nonpositive_counts(option, value, monkeypatch, capsys):
    monkeypatch.setattr(sd, "run_suite", lambda *args, **kwargs: pytest.fail("runner must not start"))
    with pytest.raises(SystemExit) as exc:
        sd.main([option, value])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert option in error
    assert "positive integer" in error


def test_cli_accepts_positive_boundaries(monkeypatch, capsys):
    calls = []

    def fake_runner(programs, seed, max_ops, limit=None):
        calls.append((programs, seed, max_ops, limit))
        return {"pass": 1, "mismatch": 0, "skip": {}, "harness-error": 0}, None

    monkeypatch.setattr(sd, "run_suite", fake_runner)
    assert sd.main(["--programs", "1", "--max-ops", "1", "--limit", "1", "--seed", "7"]) == 0
    assert calls == [(1, 7, 1, 1)]
    assert "1 pass" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["--self-test", "--list-ops"])
def test_utility_modes_skip_runner(mode, monkeypatch):
    monkeypatch.setattr(sd, "run_suite", lambda *args, **kwargs: pytest.fail("runner must not start"))
    monkeypatch.setattr(sd, "self_test", lambda: True)
    assert sd.main([mode]) == 0
