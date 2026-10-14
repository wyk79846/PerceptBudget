import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_setup


class CandidateParameterCheckTests(unittest.TestCase):
    def _config(self, temp_root, *, n=2, precision="bfloat16", state_enabled=False):
        return {
            "model_id": "baseline-candidate",
            "python_executable": "/unused/python",
            "flashrt_root": "/unused/FlashRT",
            "flashrt_revision": check_setup.FLASHRT_REVISION,
            "openpi_root": "/unused/openpi",
            "checkpoint_path": "/unused/checkpoint",
            "model_path": "/unused/checkpoint/model.safetensors",
            "norm_stats_path": "/unused/checkpoint/norm_stats.json",
            "source_document": "/unused/task.docx",
            "output_root": str(Path(temp_root) / "results"),
            "candidate_q": [5],
            "state_enabled": state_enabled,
            "H": 10,
            "N": n,
            "precision": precision,
        }

    def _parameter_results(self, config):
        with patch.object(check_setup, "check_path", return_value=None):
            results = check_setup.validate_config(config)
        return {item["name"]: item["status"] for item in results}

    def test_accepts_positive_n_and_supported_candidate_precision(self):
        with tempfile.TemporaryDirectory() as temp_root:
            statuses = self._parameter_results(self._config(temp_root))
        self.assertEqual(statuses["N"], "ok")
        self.assertEqual(statuses["precision"], "ok")
        self.assertEqual(statuses["state_enabled"], "ok")

    def test_rejects_invalid_n_or_precision(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config = self._config(temp_root, n=0, precision="fp32")
            statuses = self._parameter_results(config)
        self.assertEqual(statuses["N"], "error")
        self.assertEqual(statuses["precision"], "error")


if __name__ == "__main__":
    unittest.main()
