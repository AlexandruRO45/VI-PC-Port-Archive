"""CPU-only regression for geometry output signature register bounds."""
import pytest

import rdna2spirv
from conftest import make_dxbc, signature_chunk


@pytest.mark.parametrize("register", [1, 31, 32, 0xffffffff])
def test_gs_output_register_bound(tmp_path, monkeypatch, register):
    disasm = tmp_path / "disasm"
    disasm.mkdir()
    path = disasm / "gs_bound.s"
    path.write_text("; gs_bound synthetic\n\ts_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)\n\ts_endpgm\n", encoding="utf-8")
    dxbc = make_dxbc([signature_chunk(b"ISG1", [(0, 1, 15)]),
                      signature_chunk(b"OSG1", [(register, 0, 15)])])
    (tmp_path / "gs_bound.dxbc").write_bytes(dxbc)
    original = rdna2spirv.Gen.cu
    calls = 0

    def bounded_cu(self, value):
        nonlocal calls
        calls += 1
        assert calls < 500, "geometry setup exceeded constant budget"
        return original(self, value)

    monkeypatch.setattr(rdna2spirv.Gen, "cu", bounded_cu)
    if register >= 32:
        with pytest.raises(rdna2spirv.Unsupported, match="^gs signature$"):
            rdna2spirv.translate(str(path))
    else:
        assert "OpEmitVertex" in rdna2spirv.translate(str(path))
