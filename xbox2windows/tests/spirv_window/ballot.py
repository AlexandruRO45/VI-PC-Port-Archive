"""Exercise the named ballot profile before HLSL compilation and at GPU readback."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from spirv_ballot_extract import lower_ballot_extract


def check_ballot_profile(original):
    extract = "%bit = OpGroupNonUniformBallotBitExtract %bool %scope %mask %i"
    assert original.splitlines().count(extract) == 1
    for instruction in (extract.replace("%bool", "%uint"), extract + " %one"):
        probe = original.replace(extract, instruction)
        assert instruction in lower_ballot_extract(probe).splitlines(), instruction

    reserved = original.replace("%one", "%x2w_ballot_existing")
    try:
        lower_ballot_extract(reserved)
    except ValueError as error:
        assert str(error) == "Reserved ballot lowering ID already in use", error
    else:
        raise AssertionError("accepted reserved ballot lowering ID")


def emit(directory, assembler, validator, cross):
    source = Path(__file__).with_name("ballot.spvasm")
    original = source.read_text()
    check_ballot_profile(original)

    renamed = (original.replace("%mask", "%input_mask")
               .replace("%i =", "%lane =")
               .replace(" %i\n", " %lane\n")
               .replace("%zero", "%output_zero")
               .replace("%one", "%output_one")
               .replace("%main = OpFunction", "%main =  OpFunction"))
    variant = directory / "ballot-renamed.spvasm"
    variant.write_text(renamed)
    lowered = lower_ballot_extract(renamed)
    # The variant is valid SPIR-V even unlowered, so check that the extraction was replaced.
    assert "OpGroupNonUniformBallotBitExtract" not in lowered
    assert "OpVectorExtractDynamic %uint %input_mask" in lowered
    variant_adapted = directory / "ballot-renamed-logical.spvasm"
    variant_adapted.write_text(lowered)

    adapted = directory / "logical.spvasm"
    adapted.write_text(lower_ballot_extract(original))
    module = directory / "module.spv"
    for path in (source, variant, variant_adapted, adapted):
        subprocess.run([assembler, "--target-env", "vulkan1.2", path, "-o", module], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
    hlsl = directory / "ballot.hlsl"
    subprocess.run([cross, module, "--hlsl", "--shader-model", "66", "--output", hlsl], check=True)
    return hlsl


def draw(directory, dxil, renderer, adapter):
    mask = 0x784632A0A156789102CABEDE81234567
    bits = [(mask >> i) & 1 for i in range(128)]
    for wrong in (False, True):
        expected = bits.copy()
        if wrong:
            expected[31], expected[32] = expected[32], expected[31]
            assert expected != bits
        case = directory / "ballot.case"
        case.write_bytes(struct.pack("<321I", 4, *expected, *([0xc5c5c5c5] * 192)))
        run = subprocess.run([renderer, adapter, "--spirv-ballot", dxil, case],
                             capture_output=True, text=True)
        print(run.stdout, run.stderr, flush=True)
        if run.returncode == 77 and not wrong:
            return 77
        mismatches = 2 if wrong else 0
        if (run.returncode != int(wrong)
                or f"compared_words=320 mismatches={mismatches}" not in run.stdout):
            raise RuntimeError("Unexpected ballot readback")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        hlsl = emit(directory, *sys.argv[1:4])
        if len(sys.argv) > 4:
            dxil = directory / "ballot.dxil"
            subprocess.run([sys.argv[4], "-T", "cs_6_6", "-E", "main", "-Fo", dxil, hlsl], check=True)
            sys.exit(draw(directory, dxil, *sys.argv[5:]))
