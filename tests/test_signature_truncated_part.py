import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdna2spirv


def container(tmp_path, tag, payload):
    source_dir = tmp_path / "disasm"
    source_dir.mkdir(exist_ok=True)
    source = source_dir / "synthetic.s"
    source.touch()
    header = b"DXBC" + bytes(24) + struct.pack("<II", 1, 36)
    (tmp_path / "synthetic.dxbc").write_bytes(
        header + tag + struct.pack("<I", len(payload)) + payload
    )
    return str(source), "synthetic"


@pytest.mark.parametrize("tag,reader", [
    (b"ISG1", rdna2spirv.input_signature),
    (b"OSG1", rdna2spirv.output_signature),
    (b"OSG1", rdna2spirv.output_signature_full),
])
def test_truncated_signature_is_per_shader_failure(tmp_path, tag, reader):
    args = container(tmp_path, tag, struct.pack("<II", 1, 8) + bytes(23))
    with pytest.raises(rdna2spirv.Unsupported, match="truncated .* signature"):
        reader(*args)


@pytest.mark.parametrize("tag,reader", [
    (b"ISG1", rdna2spirv.input_signature),
    (b"OSG1", rdna2spirv.output_signature),
    (b"OSG1", rdna2spirv.output_signature_full),
])
def test_signature_shorter_than_its_header_is_per_shader_failure(tmp_path, tag, reader):
    args = container(tmp_path, tag, bytes(4))
    with pytest.raises(rdna2spirv.Unsupported, match="truncated .* signature"):
        reader(*args)


@pytest.mark.parametrize("tag,reader", [
    (b"ISG1", rdna2spirv.input_signature),
    (b"OSG1", rdna2spirv.output_signature),
    (b"OSG1", rdna2spirv.output_signature_full),
])
def test_complete_signature_keeps_register(tmp_path, tag, reader):
    element = struct.pack("<6I", 0, 0, 0, 0, 3, 7) + bytes([15, 15]) + bytes(6)
    args = container(tmp_path, tag, struct.pack("<II", 1, 8) + element)
    assert reader(*args)[0][0] == 7


def test_batch_continues_after_truncated_signature(tmp_path, monkeypatch, capsys):
    source, _ = container(tmp_path, b"ISG1", struct.pack("<II", 1, 8) + bytes(23))
    folder = Path(source).parent
    Path(source).rename(folder / "a.s")
    (tmp_path / "synthetic.dxbc").rename(tmp_path / "a.dxbc")
    (folder / "b.s").touch()
    visited = []

    def translate(path):
        name = Path(path).stem
        visited.append(name)
        if name == "a":
            rdna2spirv.input_signature(path, name)
        return "; synthetic assembly\n"

    monkeypatch.setattr(rdna2spirv, "translate", translate)
    monkeypatch.setattr(rdna2spirv, "assemble_spirv", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys, "argv", ["rdna2spirv", str(folder), "--out", str(tmp_path / "out"), "--no-validate"])
    rdna2spirv.main()
    assert visited == ["a", "b"]
    assert "1 translated" in capsys.readouterr().out
