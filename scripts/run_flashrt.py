#!/usr/bin/env python3
"""Select the configured FlashRT checkout, then optionally run a Python script."""

import argparse
import importlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolved(value):
    try:
        return Path(value or os.getcwd()).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _inside(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _finder_module_name(finder):
    return (getattr(finder, "__module__", "")
            or getattr(getattr(finder, "__func__", None), "__module__", ""))


def _foreign_flashrt_finder(finder, package_root):
    module_name = _finder_module_name(finder)
    module = sys.modules.get(module_name)
    sources = (finder, module) if module is not None else (finder,)
    locations = []
    for source in sources:
        mapping = getattr(source, "MAPPING", {}) or {}
        if "flash_rt" in mapping:
            locations.append(mapping["flash_rt"])
        namespaces = getattr(source, "NAMESPACES", {}) or {}
        for name, paths in namespaces.items():
            if name == "flash_rt" or name.startswith("flash_rt."):
                locations.extend(paths)
    if locations:
        return any(not _inside(path, package_root) for path in locations)
    return module_name.startswith("__editable___flash_rt_")


def _remove_foreign_flashrt_finders(package_root):
    for finder in tuple(sys.meta_path):
        if _foreign_flashrt_finder(finder, package_root):
            sys.meta_path.remove(finder)
    for hook in tuple(sys.path_hooks):
        if _foreign_flashrt_finder(hook, package_root):
            sys.path_hooks.remove(hook)
    for entry in tuple(sys.path):
        if str(entry).startswith("__editable__.flash_rt"):
            sys.path.remove(entry)
    for entry in tuple(sys.path_importer_cache):
        if str(entry).startswith("__editable__.flash_rt"):
            sys.path_importer_cache.pop(entry, None)


def select_flashrt(root, openpi_root=None):
    root = Path(root).expanduser().resolve()
    package_root = root / "flash_rt"
    if not (package_root / "__init__.py").is_file():
        raise FileNotFoundError("FlashRT package not found under " + str(root))
    os.environ["FLASHRT_ROOT"] = str(root)
    if openpi_root:
        os.environ["OPENPI_ROOT"] = str(Path(openpi_root).expanduser().resolve())

    loaded = sys.modules.get("flash_rt")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file and _inside(loaded_file, package_root):
            return loaded
        raise RuntimeError("flash_rt 已在本进程从其他路径导入；请用新进程启动")

    pythonpath = os.environ.get("PYTHONPATH")
    entries = pythonpath.split(os.pathsep) if pythonpath is not None else []
    blocked = set()
    retained = []
    for entry in entries:
        resolved = _resolved(entry)
        if resolved and resolved != root and (resolved / "flash_rt" / "__init__.py").is_file():
            blocked.add(resolved)
        else:
            retained.append(entry)
    if retained:
        os.environ["PYTHONPATH"] = os.pathsep.join(retained)
    else:
        os.environ.pop("PYTHONPATH", None)

    kept = []
    for entry in sys.path:
        if str(entry).startswith("__editable__.flash_rt"):
            continue
        resolved = _resolved(entry)
        if resolved in blocked:
            continue
        if resolved and resolved != root and (resolved / "flash_rt" / "__init__.py").is_file():
            continue
        kept.append(entry)
    sys.path[:] = [str(root)] + [entry for entry in kept if _resolved(entry) != root]
    for entry in tuple(sys.path_importer_cache):
        if str(entry).startswith("__editable__.flash_rt"):
            sys.path_importer_cache.pop(entry, None)
    _remove_foreign_flashrt_finders(package_root)

    spec = importlib.util.find_spec("flash_rt")
    if spec is None or not spec.origin or not _inside(spec.origin, package_root):
        origin = None if spec is None else spec.origin
        raise ImportError("flash_rt 解析到了非配置路径：" + str(origin))
    module = importlib.import_module("flash_rt")
    module_file = getattr(module, "__file__", None)
    if not module_file or not _inside(module_file, package_root):
        raise ImportError("flash_rt.__file__ 不在配置目录内：" + str(module_file))
    return module


def _configured_paths():
    config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    root = config["flashrt_root"]
    if not Path(root).is_absolute():
        raise ValueError("config.json 的 flashrt_root 必须是绝对路径")
    openpi_root = config.get("openpi_root")
    return Path(root), openpi_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="FlashRT 源码根目录；默认读取 config.json")
    parser.add_argument("--check-import", action="store_true", help="只导入并报告实际路径")
    parser.add_argument("script", nargs="?", help="要在当前进程中运行的 Python 脚本")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.check_import == bool(args.script):
        parser.error("指定 --check-import，或指定一个 Python 脚本")

    try:
        configured_root, openpi_root = _configured_paths()
        root = (args.root or configured_root).expanduser().resolve()
        module = select_flashrt(root, openpi_root)
    except ModuleNotFoundError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "error": str(exc), "missing_dependency": exc.name}, ensure_ascii=False))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False))
        return 1

    module_file = str(Path(module.__file__).resolve())
    if args.check_import:
        missing = importlib.util.find_spec("flash_rt.__perceptbudget_missing_probe_7f4a91__")
        foreign_finders = sum(
            _foreign_flashrt_finder(finder, root / "flash_rt")
            for finder in sys.meta_path
        )
        expected_openpi = str(Path(openpi_root).expanduser().resolve()) if openpi_root else None
        issues = []
        if not _inside(module_file, root / "flash_rt"):
            issues.append("flash_rt.__file__ 不在配置 root")
        if os.environ.get("FLASHRT_ROOT") != str(root):
            issues.append("FLASHRT_ROOT 未指向配置 root")
        if expected_openpi and os.environ.get("OPENPI_ROOT") != expected_openpi:
            issues.append("OPENPI_ROOT 未指向配置 root")
        if foreign_finders:
            issues.append("仍有旧 FlashRT finder")
        if missing is not None:
            issues.append("缺失子模块解析到了其他路径：" + str(missing.origin))
        print(json.dumps({
            "ok": not issues,
            "flash_rt_file": module_file,
            "FLASHRT_ROOT": os.environ.get("FLASHRT_ROOT"),
            "OPENPI_ROOT": os.environ.get("OPENPI_ROOT"),
            "foreign_flashrt_finders": foreign_finders,
            "missing_submodule_fallback": missing is None,
            "issues": issues,
        }, ensure_ascii=False))
        return 0 if not issues else 1

    script = Path(args.script).expanduser().resolve()
    if not script.is_file():
        parser.error("脚本不存在：" + str(script))
    script_dir = script.parent
    if script_dir != root:
        if not (script_dir / "flash_rt" / "__init__.py").is_file():
            sys.path.insert(1, str(script_dir))
    sys.argv = [str(script), *args.script_args]
    print("flash_rt.__file__=" + module_file, file=sys.stderr)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
