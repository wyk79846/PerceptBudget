"""Configuration and result-directory helpers for explicit experiment runs."""

import argparse
import copy
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASES = tuple("M{}".format(number) for number in range(1, 7))
_PURPOSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")
_DEPENDENCY_DIRECTORY_KEYS = ("flashrt_root", "openpi_root", "checkpoint_path")
_DEPENDENCY_FILE_KEYS = ("model_path", "norm_stats_path")


class ConfigurationError(ValueError):
    """Raised when the run configuration is missing or unsafe."""


def _resolved_repo_root(repo_root):
    root = Path(repo_root or PROJECT_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError("项目根目录不是现存目录：" + str(root))
    return root


def _output_root(config, repo_root):
    value = config.get("output_root")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("配置 output_root 必须是非空绝对路径")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigurationError("配置 output_root 必须是绝对路径")
    try:
        root = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("无法解析 output_root：" + str(exc)) from exc
    try:
        root.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ConfigurationError("output_root 必须位于 PerceptBudget 仓库之外")
    broad_roots = {Path("/").resolve(), Path.home().resolve(), repo_root.parent.resolve()}
    if any(_is_within(protected, root) for protected in broad_roots):
        raise ConfigurationError("output_root 不能是文件系统根、主目录或 workspace 等宽泛目录")
    for dependency in _dependency_directories(config, repo_root):
        if _is_within(root, dependency) or _is_within(dependency, root):
            raise ConfigurationError(
                "output_root 与固定依赖目录重叠：{}".format(dependency)
            )
    if root.exists() and not root.is_dir():
        raise ConfigurationError("output_root 已存在但不是目录：" + str(root))
    return root


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _dependency_directories(config, repo_root):
    protected = []
    for key in _DEPENDENCY_DIRECTORY_KEYS + _DEPENDENCY_FILE_KEYS + ("python_executable",):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        if key == "python_executable":
            # Capture the configured venv before resolving a python symlink to
            # its system interpreter, then protect the interpreter's real bin dir.
            protected.append(path.parent.parent.resolve())
            protected.append(path.resolve().parent)
            continue
        path = path.resolve()
        if key in _DEPENDENCY_FILE_KEYS:
            path = path.parent
        protected.append(path)
    return protected


def load_config(config_path=None, *, repo_root=None):
    """Load the project JSON config and validate its external result root."""
    root = _resolved_repo_root(repo_root)
    source = Path(config_path or root / "config.json").expanduser()
    try:
        source = source.resolve(strict=True)
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("无法读取 JSON 配置：{} ({})".format(source, exc)) from exc
    if not isinstance(config, dict):
        raise ConfigurationError("配置顶层必须是 JSON 对象")
    _output_root(config, root)
    return config


def _git_provenance(repo_root):
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False, timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            text=True, capture_output=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"git_commit": None, "git_worktree": "unavailable"}
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    worktree = (
        "clean" if not status.stdout.strip() else "dirty"
    ) if status.returncode == 0 else "unavailable"
    return {"git_commit": commit, "git_worktree": worktree}


def _write_json_atomically(path, value):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run-", suffix=".tmp", dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def create_run(config, phase, purpose="screen", *, repo_root=None):
    """Create a unique run directory and its initial provenance record."""
    if not isinstance(config, dict):
        raise ConfigurationError("配置必须是 JSON 对象")
    root = _resolved_repo_root(repo_root)
    if phase not in PHASES:
        raise ConfigurationError("phase 必须是 M1 到 M6 之一")
    if not isinstance(purpose, str) or not _PURPOSE_PATTERN.fullmatch(purpose):
        raise ConfigurationError("用途标签须为 1–32 位字母、数字、下划线或连字符")

    output_root = _output_root(config, root)
    provenance = _git_provenance(root)
    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    phase_dir = output_root / phase
    root_preexisting = output_root.exists()
    phase_created = False
    run_dir = None

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        if phase_dir.is_symlink():
            raise ConfigurationError("阶段目录不能是符号链接：" + str(phase_dir))
        try:
            phase_dir.mkdir()
            phase_created = True
        except FileExistsError:
            if not phase_dir.is_dir():
                raise ConfigurationError("阶段路径已存在但不是目录：" + str(phase_dir))

        for _ in range(10):
            run_id = "{}-{}-{}".format(timestamp, purpose.lower(), secrets.token_hex(4))
            candidate = phase_dir / run_id
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            run_dir = candidate
            break
        if run_dir is None:
            raise FileExistsError("多次生成的 run_id 均已存在")

        metadata = {
            "schema_version": 1,
            "status": "created",
            "phase": phase,
            "purpose": purpose.lower(),
            "run_id": run_dir.name,
            "created_at_utc": created_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "output_root": str(output_root),
            "config_snapshot": copy.deepcopy(config),
            **provenance,
        }
        _write_json_atomically(run_dir / "run.json", metadata)
        return run_dir
    except BaseException:
        if run_dir is not None:
            try:
                run_dir.rmdir()
            except OSError:
                pass
        if phase_created:
            try:
                phase_dir.rmdir()
            except OSError:
                pass
        if not root_preexisting:
            try:
                output_root.rmdir()
            except OSError:
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="显式创建 PerceptBudget 实验目录；不加载模型或控制设备。",
    )
    parser.add_argument("phase", choices=PHASES, help="阶段 M1–M6")
    parser.add_argument("purpose", nargs="?", default="screen", help="用途标签，默认 screen")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json",
                        help="配置文件，默认项目根目录 config.json")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run_dir = create_run(config, args.phase, args.purpose)
    except (OSError, ConfigurationError) as exc:
        print("创建失败：" + str(exc), file=sys.stderr)
        return 1
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
