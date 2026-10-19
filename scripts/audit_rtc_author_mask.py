"""Execute author's unmodified prefix function using NumPy as array backend.

Not a model/VJP equivalence test: intentionally bounded to Eq.5 only.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.rtc_guidance import soft_mask


def main():
    import numpy as np
    repo = Path('/home/wyk/workspace/任务产出/perceptbudget/rtc-author-reference')
    tree = ast.parse((repo/'src/model.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='get_prefix_weights')
    scope = dict(jnp=np, PrefixAttentionSchedule=str)
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[]))
    exec(compile(module, str(repo/'src/model.py'), 'exec'), scope)
    maximum, cases = 0., 0
    for h in (20, 50):
        for s in range(h+1):
            for d in range(h-s+1):
                reference = scope['get_prefix_weights'](d, h-s, h, 'exp')
                got = np.asarray(soft_mask(h, d, s))
                maximum = max(maximum, float(np.max(np.abs(reference-got))))
                cases += 1
    print(json.dumps(dict(scope='author Eq5 only; not independent expert or VJP validation',
        author_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        cases=cases, max_abs=maximum, passed=maximum<1e-12)))


if __name__ == '__main__':
    main()
