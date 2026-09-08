import importlib.util
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('scripts.analysis.paper_readiness')
        self.assertIsNotNone(spec, 'readiness implementation must exist')
        from scripts.analysis import paper_readiness
        self.m = paper_readiness

    def test_fallback_cost_is_not_free(self):
        self.assertAlmostEqual(self.m.corrected_saving(0, .4, .01), -.01)
        self.assertAlmostEqual(self.m.corrected_saving(.5, .4, .01), .295)

    def test_compute_mismatch_cannot_be_called_matched(self):
        self.assertFalse(self.m.compute_matched([.255, .112, .111]))
        self.assertTrue(self.m.compute_matched([.20, .202, .201]))
        self.assertFalse(self.m.compute_matched([.20, None, .201]))

    def test_latency_report_does_not_invent_acceleration(self):
        rendered = self.m.latency_readme([dict(device='cuda', dataset='cifar10', batch_size=1,
                    actual_saving_percent_mean=-7.0, actual_saving_percent_sample_sd=1.0)])
        self.assertIn('-7.000', rendered)
        self.assertIn('slowdown', rendered)

    def test_stop_on_error_and_preserve_partial_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            commands = [('first', [sys.executable, '-c', 'raise SystemExit(3)']),
                        ('never', [sys.executable, '-c', 'raise SystemExit(0)'])]
            with self.assertRaises(Exception):
                self.m.run_stages(out, commands)
            receipt = self.m.read_json(out / 'batch_status.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(len(receipt['stages']), 1)
            self.assertEqual(receipt['stages'][0]['returncode'], 3)


if __name__ == '__main__':
    unittest.main()
