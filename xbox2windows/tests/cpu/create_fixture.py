"""Generate the source-owned create translation for the Windows comparison."""
from pathlib import Path
import sys

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from ffx_create_context import create_source
from ffx_version_query import export_body


def generate(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        body_start = export_rva(pe, "create_body")
        oom = export_rva(pe, "wrapper_oom")
        done = export_rva(pe, "create_done")
        body = export_body(pe, "fixture_create", [(body_start, oom)])
        calls = {export_rva(pe, label): name for label, name in
                 (("call_provider", "create"), ("call_allocate", "allocate"), ("call_calloc", "calloc"))}
    return create_source(body, {}, calls, {done: "return static_cast<uint32_t>(rax);", oom: "return 5;"},
                         ["rcx = reinterpret_cast<word>(output); rdx = reinterpret_cast<word>(desc);",
                          "r8 = reinterpret_cast<word>(allocator); r9 = reinterpret_cast<word>(host);"])


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(generate(Path(sys.argv[1])), encoding="ascii")
