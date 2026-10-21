"""Select only the configured FlashRT checkout; no standalone legacy CLI."""
import importlib
import importlib.util
import os
import sys
from pathlib import Path


def probe_import(root, openpi_root=None):
    """Read-only import report for an isolated setup-check process."""
    try:
        root = Path(root).expanduser().resolve()
        module = select_flashrt(root, openpi_root)
        module_file = str(Path(module.__file__).resolve())
        missing = importlib.util.find_spec('flash_rt.__perceptbudget_missing_probe_7f4a91__')
        foreign = sum(_foreign_flashrt_finder(f, root / 'flash_rt') for f in sys.meta_path)
        expected = str(Path(openpi_root).expanduser().resolve()) if openpi_root else None
        issues = []
        if not _inside(module_file, root / 'flash_rt'):
            issues.append('flash_rt.__file__ is outside configured root')
        if os.environ.get('FLASHRT_ROOT') != str(root):
            issues.append('FLASHRT_ROOT mismatch')
        if expected and os.environ.get('OPENPI_ROOT') != expected:
            issues.append('OPENPI_ROOT mismatch')
        if foreign or missing is not None:
            issues.append('foreign FlashRT finder/submodule fallback')
        return dict(ok=not issues, flash_rt_file=module_file,
                    FLASHRT_ROOT=os.environ.get('FLASHRT_ROOT'),
                    OPENPI_ROOT=os.environ.get('OPENPI_ROOT'),
                    foreign_flashrt_finders=foreign,
                    missing_submodule_fallback=missing is None, issues=issues)
    except Exception as exc:
        return dict(ok=False, error_type=type(exc).__name__, error=str(exc),
                    missing_dependency=getattr(exc, 'name', None))

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
