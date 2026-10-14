import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from perceptbudget.experiment import (
    ConfigurationError,
    create_run,
    load_config,
)


class ExperimentRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp_root = Path(self.temporary.name)
        self.repo_root = self.temp_root / "project"
        self.repo_root.mkdir()
        self.output_root = self.temp_root / "results"
        self.config_path = self.temp_root / "config.json"
        self.config_data = {
            "output_root": str(self.output_root),
            "model_id": "baseline-candidate",
            "N": 1,
            "precision": "fp16",
            "baseline_status": "pending_confirmation",
        }
        with self.config_path.open("w", encoding="utf-8") as stream:
            json.dump(self.config_data, stream)

    def test_rejects_result_root_inside_repository(self):
        config = {"output_root": str(self.repo_root / "results")}
        with self.assertRaises(ConfigurationError):
            create_run(config, "M1", repo_root=self.repo_root)
        self.assertFalse((self.repo_root / "results").exists())

    def test_rejects_workspace_root_and_fixed_dependency_paths_before_creation(self):
        workspace_root = self.repo_root.parent
        with self.assertRaises(ConfigurationError):
            create_run({"output_root": str(workspace_root)}, "M1", repo_root=self.repo_root)
        dependency = self.temp_root / "FlashRT-upstream"
        config = {
            "output_root": str(dependency / "results"),
            "flashrt_root": str(dependency),
        }
        with self.assertRaises(ConfigurationError):
            create_run(config, "M1", repo_root=self.repo_root)
        self.assertFalse(dependency.exists())

    def test_rejects_output_inside_configured_venv_even_when_python_is_a_symlink(self):
        venv = self.temp_root / "task-venv"
        python_bin = venv / "bin" / "python"
        actual_bin = self.temp_root / "system-bin"
        python_bin.parent.mkdir(parents=True)
        actual_bin.mkdir()
        actual_python = actual_bin / "python3"
        actual_python.touch()
        python_bin.symlink_to(actual_python)
        config = {
            "output_root": str(venv / "results"),
            "python_executable": str(python_bin),
        }
        with self.assertRaises(ConfigurationError):
            create_run(config, "M1", repo_root=self.repo_root)
        self.assertFalse((venv / "results").exists())

    def test_rejects_path_traversal_labels(self):
        config = {"output_root": str(self.output_root)}
        with self.assertRaises(ConfigurationError):
            create_run(config, "../M1", repo_root=self.repo_root)
        with self.assertRaises(ConfigurationError):
            create_run(config, "M1", "../screen", repo_root=self.repo_root)
        self.assertFalse(self.output_root.exists())

    def test_runs_are_independent_and_do_not_overwrite(self):
        config = load_config(self.config_path, repo_root=self.repo_root)
        first = create_run(config, "M1", "screen", repo_root=self.repo_root)
        second = create_run(config, "M1", "screen", repo_root=self.repo_root)
        self.assertNotEqual(first, second)
        self.assertTrue((first / "run.json").is_file())
        self.assertTrue((second / "run.json").is_file())
        self.assertEqual(len(list((self.output_root / "M1").iterdir())), 2)

    def test_metadata_contains_config_snapshot_and_provenance(self):
        config = load_config(self.config_path, repo_root=self.repo_root)
        run_dir = create_run(config, "M2", "confirm", repo_root=self.repo_root)
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "created")
        self.assertEqual(metadata["phase"], "M2")
        self.assertEqual(metadata["purpose"], "confirm")
        self.assertEqual(metadata["config_snapshot"], self.config_data)
        self.assertEqual(metadata["output_root"], str(self.output_root))
        self.assertTrue(metadata["created_at_utc"].endswith("Z"))
        self.assertIn("git_commit", metadata)
        self.assertIn("git_worktree", metadata)

    def test_metadata_write_failure_leaves_no_run(self):
        self.output_root.mkdir()
        config = {"output_root": str(self.output_root)}
        with patch("perceptbudget.experiment.json.dump", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                create_run(config, "M1", "screen", repo_root=self.repo_root)
        self.assertFalse((self.output_root / "M1").exists())


if __name__ == "__main__":
    unittest.main()
