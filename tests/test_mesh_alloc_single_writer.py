import sys
import os
import unittest

sys.path.insert(0, 'tools')
import rdna2spirv
import spirv_asm

class MeshAllocSingleWriterTests(unittest.TestCase):
    def test_ms_alloc_req_single_writer_guard(self):
        """SHADER-766: s_sendmsg(MSG_GS_ALLOC_REQ) restricts OpStore %ms_counts to the wave's lowest active lane (not local_index == 0:
        ms_wave_info_s3 sends the allocation from wave 1 only)."""
        for mesh_compute in (False, True):
            orig_mc = rdna2spirv.MESH_COMPUTE
            try:
                rdna2spirv.MESH_COMPUTE = mesh_compute
                gen = rdna2spirv.Gen("ms")
                gen.ms = {"threads": 128, "maxv": 64, "maxp": 64, "clip": 0, "cull": 0, "params": []}
                # Emit MSG_GS_ALLOC_REQ
                ins = {
                    "op": "s_sendmsg",
                    "args": [{"k": "special", "t": "sendmsg(MSG_GS_ALLOC_REQ)"}],
                    "mods": []
                }
                gen._insn(ins)

                body_text = "\n".join(gen.body)
                # 1. OpStore %ms_counts must be present
                self.assertIn("OpStore %ms_counts", body_text)

                # 2. The guard is the exec bit and the lowest active lane, never local_index
                self.assertNotIn("%local_index", body_text)
                self.assertRegex(body_text, r"%(\w+) = OpLoad %bool %active")
                self.assertIn("OpGroupNonUniformBallotFindLSB", body_text)
                self.assertIn("= OpLoad %uint %sg_lane", body_text)
                self.assertRegex(body_text, r"= OpLogicalAnd %bool %(\w+) %(\w+)")

                # 3. Must have structured selection merge and conditional branch
                self.assertRegex(body_text, r"OpSelectionMerge %(\w+) None\s+OpBranchConditional %(\w+) %(\w+) %(\w+)")

                # 4. OpStore %ms_counts must only appear inside the conditional block
                lines = [l.strip() for l in gen.body if l.strip()]
                store_idx = next(i for i, l in enumerate(lines) if "OpStore %ms_counts" in l)
                branch_cond_idx = next(i for i, l in enumerate(lines) if "OpBranchConditional" in l)
                self.assertGreater(store_idx, branch_cond_idx)

                # 5. Verify assembly into a complete module passes spirv_asm
                # Build mock header
                types_and_vars = [
                    "OpCapability Shader",
                    "OpCapability GroupNonUniformBallot",
                    "OpCapability MeshShadingEXT" if not mesh_compute else "",
                    "OpMemoryModel Logical GLSL450",
                    'OpEntryPoint GLCompute %main "main" %local_index %sg_lane' if mesh_compute else 'OpEntryPoint MeshEXT %main "main" %local_index %sg_lane',
                    "OpExecutionMode %main LocalSize 128 1 1",
                    "OpDecorate %local_index BuiltIn LocalInvocationIndex",
                    "OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId",
                    "%void = OpTypeVoid",
                    "%bool = OpTypeBool",
                    "%uint = OpTypeInt 32 0",
                    "%_ptr_Input_uint = OpTypePointer Input %uint",
                    "%c3 = OpConstant %uint 3",
                    "%v4uint_bal = OpTypeVector %uint 4",
                    "%_ptr_Private_bool = OpTypePointer Private %bool",
                    "%active = OpVariable %_ptr_Private_bool Private",
                    "%sg_lane = OpVariable %_ptr_Input_uint Input",
                    "%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint",
                    "%_ptr_Function_uint = OpTypePointer Function %uint",
                    "%local_index = OpVariable %_ptr_Input_uint Input",
                    "%ms_counts = OpVariable %_ptr_Workgroup_uint Workgroup",
                    "%m0 = OpVariable %_ptr_Function_uint Function",
                    "%fn_t = OpTypeFunction %void",
                    "%main = OpFunction %void None %fn_t",
                    "%entry = OpLabel"
                ]
                spv_text = "\n".join([line for line in types_and_vars if line]) + "\n" + body_text + "\nOpReturn\nOpFunctionEnd\n"
                bin_data = spirv_asm.assemble(spv_text)
                self.assertGreater(len(bin_data), 0)
            finally:
                rdna2spirv.MESH_COMPUTE = orig_mc

if __name__ == '__main__':
    unittest.main()
