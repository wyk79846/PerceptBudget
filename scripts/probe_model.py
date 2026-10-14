#!/usr/bin/env python3
"""Check native modules before any expensive checkpoint allocation."""

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run, load_config
from scripts.run_flashrt import select_flashrt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true",
                        help="Also load the checkpoint; no robot inference")
    args = parser.parse_args()
    config = load_config()
    run_dir = create_run(config, "M1", "backend-check")
    print("run_dir=" + str(run_dir), flush=True)
    report = {"status": "checking", "horizon_requested": config["H"],
              "inference_verified": False}
    try:
        select_flashrt(config["flashrt_root"], config["openpi_root"])
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            raise RuntimeError("CUDA is not available in this process")
        report["device"] = torch.cuda.get_device_name(0)
        report["native_modules"] = {}
        for name in ("flash_rt.flash_rt_kernels", "flash_rt.flash_rt_fa2"):
            module = importlib.import_module(name)
            report["native_modules"][name] = module.__file__
        if args.load:
            from perceptbudget.model import load_model
            start = time.monotonic()
            model = load_model(config)
            report["load_seconds"] = time.monotonic() - start
            report["frontend_horizon"] = model._pipe.chunk_size
        report["status"] = "passed_load" if args.load else "passed_import"
    except Exception as exc:
        report.update(status="invalid", error=str(exc), traceback=traceback.format_exc())
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(status=report["status"], inference_verified=False)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
