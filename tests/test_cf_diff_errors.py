"""Control-flow differential harness errors must not become successful skips."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import cf_diff


ERRORS = (
    ("empty", "listing has no instructions"),
    ("missing_end", "branch from pc 0x0 to 0x4, which is not an instruction"),
    ("bad_branch", "which is not an instruction"),
    ("step_limit", "step limit 1 reached"),
)


def fail_second_reference(monkeypatch, case):
    run = cf_diff.cf_ref.run
    calls = []

    def run_reference(path, **kwargs):
        calls.append(kwargs["wave"])
        if len(calls) == 2:
            listing = Path(path)
            if case == "step_limit":
                kwargs["max_steps"] = 1
            elif case == "unsupported":
                raise cf_diff.cpu_ref.Unsupported("synthetic opcode")
            else:
                lines = [] if case == "empty" else [("s_mov_b32 s0, 1", 4)]
                if case == "bad_branch":
                    lines = [("s_branch 100", 4), ("s_endpgm", 4)]
                listing.write_text(cf_diff.listing_text(lines, "ps_bad"), encoding="utf-8", newline="\n")
        return run(path, **kwargs)

    monkeypatch.setattr(cf_diff.cf_ref, "run", run_reference)
    return calls


@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("case, diagnostic", ERRORS)
def test_reference_error_next_to_pass_is_harness_error(wave, case, diagnostic, monkeypatch, capsys):
    calls = fail_second_reference(monkeypatch, case)
    code = cf_diff.main(["--programs", "2", "--seed", "17", "--wave", str(wave)])
    out = capsys.readouterr().out
    assert code == 2, out
    assert calls == [wave, wave]
    assert "1 pass, 0 mismatch, 0 skipped, 1 harness errors" in out
    assert f"program 1 wave{wave}: harness error ProgramError:" in out
    assert diagnostic in out


@pytest.mark.parametrize("case, diagnostic", ERRORS)
def test_replay_reference_error_after_passing_wave_exits_2(case, diagnostic, monkeypatch, capsys):
    calls = fail_second_reference(monkeypatch, case)
    code = cf_diff.main(["--program", "1", "--seed", "17", "--wave", "both"])
    out = capsys.readouterr().out
    assert code == 2, out
    assert calls == [32, 64]
    assert "program 1 wave32: match" in out
    assert "program 1 wave64: harness error ProgramError:" in out
    assert diagnostic in out


@pytest.mark.parametrize("args", (
    ["--programs", "2", "--wave", "32"],
    ["--programs", "2", "--wave", "64"],
    ["--program", "1", "--wave", "both"],
))
def test_genuine_unsupported_next_to_pass_remains_skip(args, monkeypatch, capsys):
    calls = fail_second_reference(monkeypatch, "unsupported")
    code = cf_diff.main(args + ["--seed", "17"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert len(calls) == 2
    assert "cf_ref: unsupported instruction synthetic opcode" in out
    assert "harness error ProgramError" not in out
