import unittest
from tools.check_d3d12_regressions import validate,DIGEST

class D3D12RegressionTests(unittest.TestCase):
    def test_adapter_and_exit_are_checked(self):
        line="d3d12-indirect PASS pci=10de:2704 list=59 words=64/64"
        self.assertTrue(validate("indirect","10de:2704",0,line)[0])
        self.assertFalse(validate("indirect","1002:1234",0,line)[0])
        self.assertFalse(validate("indirect","10de:2704",1,line)[0])
        self.assertFalse(validate("indirect","10de:2704",0,line+"\n"+line)[0])
        self.assertFalse(validate("indirect-scarlett","10de:2704",0,line)[0])
    def test_null_checkpoint_must_match(self):
        line="d3d12-null-usage PASS pci=10de:2704 words=64/64 recorded=64 valid=2"
        logs=[line]
        for interval,count in ((1,1),(2,63)):
            logs.append(f"TuxBox d3d12_xs: null usage interval recorded={count} display_bound=0 tracked=1 overflow=0 overflow_display_bound=0 compute={count} raster=0 interval={interval} trigger=recording_checkpoint")
            logs.append(f"TuxBox d3d12_xs: null usage rank kind=compute CS={DIGEST} recorded={count} display_bound=0 interval={interval} trigger=recording_checkpoint")
        output="\n".join(logs)
        self.assertTrue(validate("null-usage","10de:2704",0,output)[0])
        self.assertFalse(validate("null-usage","10de:2704",0,output.replace("compute=63","compute=64"))[0])
        self.assertFalse(validate("null-usage","10de:2704",0,line)[0])

    def test_buffer_import_requires_exact_backing_counts(self):
        line="d3d12-buffer-import PASS pci=10de:2704 pages=2 words=1536/1536 passes=2 aliases=2"
        records=[("pool_open",i) for i in range(1,4)]+[("pool_hit",i) for i in range(1,3)]+[("imported",i) for i in range(1,6)]
        logs=[line]+[f"TuxBox d3d12_xs: buffer backing path={p} count={i} source=place_virtual bytes=65536" for p,i in records]
        self.assertTrue(validate("buffer-import","10de:2704",0,"\n".join(logs))[0])
        self.assertFalse(validate("buffer-import","10de:2704",0,"\n".join(logs[:-1]))[0])
        self.assertFalse(validate("buffer-import","10de:2704",0,"\n".join(logs+[
            "TuxBox d3d12_xs: buffer backing path=device_standin count=1 source=place_virtual bytes=65536"]))[0])

    def test_late_commit_checks_both_imports_and_no_standin(self):
        line="d3d12-buffer-layout PASS pci=10de:2704 slot=133 words=512/512 VA-offsets=0,4"
        logs=[line]+[f"TuxBox d3d12_xs: buffer backing path={p} count={i} source=place_virtual_layout bytes={1024+4*(i-1)}"
                     for p in ("imported","commit_retry") for i in (1,2)]
        self.assertTrue(validate("buffer-layout","10de:2704",0,"\n".join(logs))[0])
        self.assertFalse(validate("buffer-layout","10de:2704",0,"\n".join(logs[:-1]))[0])
        self.assertFalse(validate("buffer-layout","10de:2704",0,"\n".join(logs).replace("commit_retry","device_standin"))[0])
        negative="d3d12-buffer-import-reject PASS pci=10de:2704 rejected=1"
        record="TuxBox d3d12_xs: buffer backing path=strict_rejected count=1 source=place_virtual bytes=65536"
        self.assertTrue(validate("buffer-import-reject","10de:2704",0,negative+"\n"+record)[0])
        self.assertFalse(validate("buffer-import-reject","10de:2704",0,negative)[0])
