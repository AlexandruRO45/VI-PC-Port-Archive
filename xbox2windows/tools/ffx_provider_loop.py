"""Bounded default-provider create path for the pinned upscaler image."""
import struct

from ffx_version_query import export_body, leaf, lower, region


def generate_loop(pe):
    def blocks(*ranges):
        return [ins for start, end in ranges for ins in region(pe, start, end)]

    source = ['#include "ffx_provider_loop.h"']
    body = region(pe, 0xbf95, 0xc750)
    globals_ = {}
    for addr in sorted({i.ip_rel_memory_address for i in body if i.is_ip_rel_memory_operand}):
        symbol = f"loop_data_{addr:x}"
        if addr in (0x486f0, 0x48838, 0xb79d0):
            count = 4 if addr == 0xb79d0 else 2
            values = struct.unpack(f"<{count}I", pe.get_data(addr, count * 4))
            source.append(f"static const uint32_t {symbol}[] = {{{', '.join(map(str, values))}}};")
        else:
            name = pe.get_string_u_at_rva(addr).decode("ascii")
            if not name or not name.replace("_", "").isalnum() or not name.isascii():
                raise ValueError("unexpected provider resource name")
            source.append(f'static const wchar_t {symbol}[] = L"{name}";')
        globals_[addr] = f"reinterpret_cast<word>({symbol})"
    size = lower(leaf(pe, 0x7900), ["rdx = tier;"], {}, {}, {})
    source.append("static word scratch_size(word tier) {\n" + size + "\n}")
    helper_size = region(pe, 0x1049, 0x1054)[0].immediate(1)
    setup = blocks((0xcfcb, 0xcfd2), (0xcfe0, 0xd001), (0xd061, 0xd064),
                   (0xd07c, 0xd083), (0xd094, 0xd098), (0xd0f5, 0xd0fc))
    build = lower(setup + body,
                  ["uint8_t local[192]{}; rbp = reinterpret_cast<word>(local + 112);",
                   "rdi = reinterpret_cast<word>(&ctx.state); rdx = reinterpret_cast<word>(&config);",
                   f"word helper[] = {{0, 0, {helper_size}}}; rsi = reinterpret_cast<word>(helper);"],
                  globals_, {0xc4ff: "rax = scratch_size(rdx);"},
                  {0xcfd2: "goto block_cfe0;", 0xd001: "goto block_d061;", 0xd064: "goto block_d07c;",
                   0xd083: "goto block_d094;", 0xd098: "goto block_d0f5;",
                   0xd0fc: "r14 = r8; rbx = rdi + 0x228; rdi = reinterpret_cast<word>(descriptors); goto block_bf95;",
                   0xc750: "return;"})
    source.append("static void build(LoopContext& ctx, const RecurrentConfig& config, LoopDesc* descriptors) {\n" + build + "\n}")
    allocation = lower(region(pe, 0x2ee6, 0x2f1f),
                       ["r12 = reinterpret_cast<word>(slot); rbx = reinterpret_cast<word>(scratch);",
                        "rsi = reinterpret_cast<word>(index); rdi = reinterpret_cast<word>(&desc); r8 = 1;"],
                       {}, {}, {0x2f1f: "return;"})
    source.append("void reserve_resource(uint8_t* slot, uint8_t* scratch, const LoopDesc& desc, uint32_t* index) {\n" + allocation + "\n}")
    upload = lower(region(pe, 0x38a8, 0x3915),
                   ["uint8_t local[320]{}; rbp = reinterpret_cast<word>(local + 128);",
                    "rdi = reinterpret_cast<word>(&desc);"], {},
                   {0x3912: "return ctx.make_upload(desc, load<LoopDesc>(rdx));"},
                   {0x39a0: "return 0;", 0x3915: "return 0;"})
    source.append("uint32_t schedule_upload(LoopContext& ctx, const LoopDesc& desc) {\n" + upload + "\n}")
    loop = lower(region(pe, 0xd117, 0xd26d) + region(pe, 0xcf52, 0xcf54),
                 ["alignas(16) uint8_t stack[0x600]{}; rsp = reinterpret_cast<word>(stack); rbp = rsp + 0x100;",
                  "rdi = reinterpret_cast<word>(&ctx.state); rsi = 3; store(rsp + 0x48, output);",
                  "LoopDesc descriptors[12]{}; build(ctx, config, descriptors); memcpy(stack+0xc0, descriptors, sizeof descriptors);"],
                 {0x2e30: "loop_callback"},
                 {0xd207: "rax = ctx.create(rcx, load<LoopDesc>(rdx), r8, reinterpret_cast<uint32_t*>(r9));",
                  0xd20e: 'throw std::runtime_error("Unsupported provider callback");'},
                 {0xcf54: "return static_cast<uint32_t>(rax);"})
    source.append("static uint32_t run_loop(LoopContext& ctx, void** output, const RecurrentConfig& config) {\n" + loop + "\n}")
    wrapper = lower(export_body(pe, "ffxCreateContext", [(0x11f0, 0x11f7), (0x1338, 0x1359), (0x1388, 0x138f)]),
                    ["word selected[] = {0,1}; rbx = reinterpret_cast<word>(selected);",
                     "r13 = rcx = reinterpret_cast<word>(output); r12 = reinterpret_cast<word>(&config);"],
                    {}, {0x1349: "rax = run_loop(ctx, reinterpret_cast<void**>(rdx), *reinterpret_cast<const RecurrentConfig*>(r8));"},
                    {0x11f7: "goto block_1338;", 0x12bc: "return static_cast<uint32_t>(rax);",
                     0x1359: 'throw std::runtime_error("Unsupported provider fallback");'})
    source.append("uint32_t create_loop_context(LoopContext& ctx, void** output, const RecurrentConfig& config) {\n" + wrapper + "\n}")
    return "\n".join(source)
