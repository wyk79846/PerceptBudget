import json
import os
import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRT_ROOT = Path(
    json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))[
        "flashrt_root"
    ]
)
TOKENIZER_PATH = FLASHRT_ROOT / "flash_rt" / "utils" / "paligemma_tokenizer.py"
TOKENIZER_SPEC = spec_from_file_location(
    "paligemma_tokenizer_under_test", TOKENIZER_PATH
)
if TOKENIZER_SPEC is None or TOKENIZER_SPEC.loader is None:
    raise ImportError(f"Cannot load tokenizer module from {TOKENIZER_PATH}")
TOKENIZER = module_from_spec(TOKENIZER_SPEC)
TOKENIZER_SPEC.loader.exec_module(TOKENIZER)


class PaligemmaTokenizerCacheTests(unittest.TestCase):
    def setUp(self):
        TOKENIZER._load_sentencepiece_cached.cache_clear()
        self.load_calls = []
        self.load_results = []

        test_case = self

        class FakeSentencePieceProcessor:
            def Load(self, path):
                test_case.load_calls.append(path)
                if test_case.load_results:
                    return test_case.load_results.pop(0)
                return True

        self.sentencepiece_patch = patch.dict(
            sys.modules,
            {
                "sentencepiece": SimpleNamespace(
                    SentencePieceProcessor=FakeSentencePieceProcessor
                )
            },
        )
        self.sentencepiece_patch.start()
        self.addCleanup(self.sentencepiece_patch.stop)

    def tearDown(self):
        TOKENIZER._load_sentencepiece_cached.cache_clear()

    def _model_file(self, directory, name, contents=b"model"):
        path = Path(directory) / name
        path.write_bytes(contents)
        return path

    def test_same_path_loads_once_and_resolves_path_each_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._model_file(temp_dir, "paligemma.model")
            with patch.object(
                TOKENIZER, "resolve_paligemma_tokenizer_path", return_value=str(path)
            ) as resolver:
                first = TOKENIZER.load_paligemma_sentencepiece()
                second = TOKENIZER.load_paligemma_sentencepiece()

        self.assertIs(first, second)
        self.assertEqual(self.load_calls, [str(path.resolve())])
        self.assertEqual(resolver.call_count, 2)

    def test_cache_key_tracks_path_mtime_and_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = self._model_file(temp_dir, "first.model")
            second_path = self._model_file(temp_dir, "second.model")
            fixed_mtime_ns = 1_700_000_000_000_000_000
            for path in (first_path, second_path):
                os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))

            with patch.object(
                TOKENIZER,
                "resolve_paligemma_tokenizer_path",
                side_effect=[
                    str(first_path),
                    str(second_path),
                    str(first_path),
                    str(first_path),
                    str(first_path),
                ],
            ) as resolver:
                first = TOKENIZER.load_paligemma_sentencepiece()
                different_path = TOKENIZER.load_paligemma_sentencepiece()
                same_metadata = TOKENIZER.load_paligemma_sentencepiece()

                stat = first_path.stat()
                changed_mtime_ns = stat.st_mtime_ns + 1_000_000_000
                os.utime(first_path, ns=(stat.st_atime_ns, changed_mtime_ns))
                changed_mtime = TOKENIZER.load_paligemma_sentencepiece()

                stat = first_path.stat()
                first_path.write_bytes(b"model with a different size")
                os.utime(first_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                changed_size = TOKENIZER.load_paligemma_sentencepiece()
                mtime_after_size_change = first_path.stat().st_mtime_ns
                size_after_change = first_path.stat().st_size

        self.assertEqual(mtime_after_size_change, changed_mtime_ns)
        self.assertNotEqual(size_after_change, len(b"model"))
        self.assertEqual(resolver.call_count, 5)
        self.assertEqual(len(self.load_calls), 4)
        self.assertIs(first, same_metadata)
        self.assertIsNot(first, different_path)
        self.assertIsNot(first, changed_mtime)
        self.assertIsNot(changed_mtime, changed_size)

    def test_failed_load_is_retried_instead_of_cached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._model_file(temp_dir, "paligemma.model")
            self.load_results.extend([False, True])
            with patch.object(
                TOKENIZER, "resolve_paligemma_tokenizer_path", return_value=str(path)
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to load"):
                    TOKENIZER.load_paligemma_sentencepiece()
                loaded = TOKENIZER.load_paligemma_sentencepiece()

        self.assertIsNotNone(loaded)
        self.assertEqual(self.load_calls, [str(path.resolve()), str(path.resolve())])


if __name__ == "__main__":
    unittest.main()
