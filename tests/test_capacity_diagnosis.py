import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io


class CapacityComparisonTests(unittest.TestCase):
    def test_bf16_bits_and_repeat_control(self):
        import numpy as np
        from scripts.diagnose_state_capacity import compare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for label,last in (('capacity200a',0x4000),('capacity64',0x4001),('capacity200b',0x4000)):
                p=root/label
                p.mkdir()
                np.savez(p/'trace.npz',mock=np.asarray([0x3f80,last],dtype=np.uint16))
                (p/'summary.json').write_text(json.dumps({'stage_order':['mock'],'status':'completed'}))
            with contextlib.redirect_stdout(io.StringIO()): compare(root)
            result=json.loads((root/'summary.json').read_text())
            diff=result['comparisons']['capacity_change']['first_difference']
            self.assertEqual(diff['stage'],'mock')
            self.assertEqual(diff['max_abs'],0.015625)
            self.assertEqual(diff['changed_fraction'],0.5)
            self.assertIsNone(result['comparisons']['same_capacity_repeat']['first_difference'])
