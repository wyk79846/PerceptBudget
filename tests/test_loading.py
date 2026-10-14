import ast
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = (
    Path(json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))["flashrt_root"])
    / "flash_rt"
    / "frontends"
    / "torch"
    / "pi05_rtx.py"
)


def _load_stack_and_release(torch):
    tree = ast.parse(FRONTEND_PATH.read_text(encoding="utf-8"), filename=str(FRONTEND_PATH))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stack_and_release"
    ]
    if len(matches) != 1:
        raise AssertionError("Expected exactly one _stack_and_release definition")

    module = ast.Module(body=matches, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(FRONTEND_PATH), "exec"), namespace)
    return namespace["_stack_and_release"]


class StackAndReleaseTests(unittest.TestCase):
    def test_bf16_noncontiguous_stack_is_bit_equal_and_clears_list(self):
        import torch

        stack_and_release = _load_stack_and_release(torch)
        source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(torch.bfloat16)
        layers = [source[index].transpose(0, 1) for index in range(2)]
        self.assertTrue(all(not layer.is_contiguous() for layer in layers))

        expected = torch.stack(layers)
        layers_to_release = list(layers)
        actual = stack_and_release(layers_to_release)

        self.assertEqual(layers_to_release, [])
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
