#!/usr/bin/env python3
"""离线汇总一次 M1.2 推理机会分析；不访问设备或模型。"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perceptbudget.power import PowerReading, _integrate_report


MAX_GAP_SECONDS = 0.3  # PowerMonitor 默认 0.1 s 采样间隔的三倍


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _merge(intervals, start, stop):
    clipped = sorted((max(start, a), min(stop, b)) for a, b in intervals if b > a)
    merged = []
    for a, b in clipped:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _integral(readings, intervals):
    duration = sum(b - a for a, b in intervals)
    if duration <= 0:
        return {"duration_seconds": 0.0, "coverage_seconds": 0.0,
                "coverage_fraction": None, "covered_energy_j": 0.0,
                "total_energy_j": 0.0, "mean_power_w": None,
                "full_coverage": True, "missing_count": 0}
    reports = [_integrate_report(readings, a, b, MAX_GAP_SECONDS)
               for a, b in intervals]
    covered = sum(report.coverage_seconds for report in reports)
    partial_mj = sum(report.energy_mj or 0.0 for report in reports)
    full = all(report.total_energy_mj is not None for report in reports)
    total_mj = partial_mj if full else None
    return {
        "duration_seconds": duration,
        "coverage_seconds": covered,
        "coverage_fraction": covered / duration,
        "covered_energy_j": partial_mj / 1000.0 if covered else None,
        "total_energy_j": total_mj / 1000.0 if total_mj is not None else None,
        "mean_power_w": total_mj / duration / 1000.0
        if total_mj is not None else None,
        "full_coverage": full,
        "missing_count": sum(report.missing_count for report in reports),
    }


def _episode(source):
    stop = _number(source.get("completed_at_monotonic"))
    elapsed = _number(source.get("real_task_seconds"))
    valid_window = stop is not None and elapsed is not None and elapsed > 0
    start = stop - elapsed if valid_window else None
    if valid_window and start < 0:
        valid_window = False

    raw = (source.get("power") or {}).get("raw_readings", [])
    readings, bad_readings = [], 0
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or _number(item.get("timestamp_monotonic")) is None:
                bad_readings += 1
                continue
            power = _number(item.get("power_mw"))
            error = item.get("error")
            readings.append(PowerReading(
                _number(item["timestamp_monotonic"]), power,
                str(error) if error else ("invalid power_mw" if power is None else None),
            ))

    requests = (source.get("inference") or {}).get("requests")
    timeline_complete = isinstance(requests, list)
    intervals = []
    for request in requests if isinstance(requests, list) else []:
        a = _number(request.get("inference_started_monotonic")) if isinstance(request, dict) else None
        b = _number(request.get("inference_ended_monotonic")) if isinstance(request, dict) else None
        if a is None or b is None or b < a:
            timeline_complete = False
            continue
        if valid_window:
            intervals.append((a, b))

    busy = _merge(intervals, start, stop) if valid_window else []
    idle = []
    if valid_window and timeline_complete:
        cursor = start
        for a, b in busy:
            if a > cursor:
                idle.append([cursor, a])
            cursor = max(cursor, b)
        if cursor < stop:
            idle.append([cursor, stop])

    busy_power = _integral(readings, busy) if valid_window else None
    idle_power = (_integral(readings, idle)
                  if valid_window and timeline_complete else None)
    d = (busy_power["duration_seconds"] / elapsed
         if valid_window and timeline_complete and elapsed else None)
    pon = busy_power["mean_power_w"] if timeline_complete and busy_power else None
    pidle = idle_power["mean_power_w"] if idle_power else None
    ratio = pidle / pon if pidle is not None and pon not in (None, 0.0) else None
    score = (1.0 - d) * (1.0 - ratio) if d is not None and ratio is not None else None

    frequency = _number(source.get("control_freq_hz"))
    buffer = (source.get("inference") or {}).get("buffer_summary_actions") or {}
    buffer_seconds = {
        key: (_number(buffer.get(key)) / frequency
              if frequency and _number(buffer.get(key)) is not None else None)
        for key in ("min", "median", "max")
    }
    complete = (valid_window and timeline_complete
                and source.get("status") in ("success", "failure", "timeout")
                and busy_power is not None
                and idle_power is not None and busy_power["full_coverage"]
                and idle_power["full_coverage"])
    return {
        "task_id": source.get("task_id"), "task_name": source.get("task_name"),
        "q": source.get("q"), "source_status": source.get("status"),
        "status": "complete" if complete else "partial",
        "window": {"start_monotonic": start, "stop_monotonic": stop,
                   "duration_seconds": elapsed if valid_window else None},
        "inference_timeline_complete": timeline_complete,
        "busy_intervals": busy if timeline_complete else None,
        "busy_seconds_known": busy_power["duration_seconds"] if busy_power else None,
        "idle_seconds": idle_power["duration_seconds"] if idle_power else None,
        "d": d, "P_on_w": pon, "P_idle_w": pidle, "r": ratio, "S": score,
        "busy_power": busy_power, "idle_power": idle_power,
        "buffer_summary_actions": buffer,
        "buffer_summary_seconds": buffer_seconds,
        "bad_power_readings_ignored": bad_readings,
    }


def _sum_covered(episodes, zone):
    values = [e[zone + "_power"]["covered_energy_j"] for e in episodes
              if e.get(zone + "_power") and e[zone + "_power"]["covered_energy_j"] is not None]
    return sum(values) if values else None


def _coverage(episodes, zone):
    reports = [e.get(zone + "_power") for e in episodes]
    if not reports or any(report is None for report in reports):
        return None
    duration = sum(report["duration_seconds"] for report in reports)
    return (sum(report["coverage_seconds"] for report in reports) / duration
            if duration else None)


def _aggregate(q, episodes, expected):
    timed = [e for e in episodes if e["window"]["duration_seconds"] is not None]
    known_timeline = (len(timed) == len(episodes)
                      and all(e["inference_timeline_complete"] for e in episodes))
    count_complete = expected is not None and len(episodes) == expected
    complete = (count_complete and known_timeline
                and all(e["status"] == "complete" for e in episodes))
    busy_s = sum(e["busy_power"]["duration_seconds"] for e in episodes
                 if e.get("busy_power"))
    total_s = sum(e["window"]["duration_seconds"] for e in timed)
    observed_d = busy_s / total_s if known_timeline and total_s else None
    busy_j = sum(e["busy_power"]["total_energy_j"] or 0.0 for e in episodes
                 if e.get("busy_power")) if complete else None
    idle_j = sum(e["idle_power"]["total_energy_j"] or 0.0 for e in episodes
                 if e.get("idle_power")) if complete else None
    busy_duration = sum(e["busy_power"]["duration_seconds"] for e in episodes
                        if e.get("busy_power"))
    idle_duration = sum(e["idle_power"]["duration_seconds"] for e in episodes
                        if e.get("idle_power"))
    pon = busy_j / busy_duration if complete and busy_duration else None
    pidle = idle_j / idle_duration if complete and idle_duration else None
    d = observed_d if complete else None
    ratio = pidle / pon if pidle is not None and pon not in (None, 0.0) else None
    return {
        "q": q, "status": "complete" if complete else "partial",
        "episodes": len(episodes), "expected_episodes": expected,
        "observed_d": observed_d, "d": d,
        "observed_busy_coverage_fraction": _coverage(episodes, "busy"),
        "observed_idle_coverage_fraction": _coverage(episodes, "idle"),
        "busy_covered_energy_j": _sum_covered(episodes, "busy"),
        "idle_covered_energy_j": _sum_covered(episodes, "idle"),
        "busy_energy_j": busy_j, "idle_energy_j": idle_j,
        "P_on_w": pon, "P_idle_w": pidle, "r": ratio,
        "S": (1.0 - d) * (1.0 - ratio)
        if d is not None and ratio is not None else None,
    }


def analyze_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
        raise ValueError("summary.json 须含 episodes 列表")
    episodes = [_episode(e) for e in summary["episodes"] if isinstance(e, dict)]
    expected = summary.get("expected_episodes")
    expected = expected if isinstance(expected, int) and not isinstance(expected, bool) else None
    run_complete = summary.get("status") == "completed" and expected == len(episodes)
    task_ids = summary.get("task_ids")
    expected_per_q = len(task_ids) if isinstance(task_ids, list) else None
    q_values = summary.get("q_values") or list(dict.fromkeys(e["q"] for e in episodes))
    q_summaries = [_aggregate(q, [e for e in episodes if e["q"] == q],
                              expected_per_q) for q in q_values]
    run_complete = (run_complete and all(e["status"] == "complete" for e in episodes)
                    and all(q["status"] == "complete" for q in q_summaries))
    result = {
        "schema_version": 1, "run_dir": str(run_dir),
        "source_status": summary.get("status"),
        "status": "complete" if run_complete else "partial",
        "expected_episodes": expected, "observed_episodes": len(episodes),
        "power_rail": "VDD_IN module input; includes rendering/control loads",
        "idle_definition": "same-run complement of inference intervals; rendering/control continue; device clocks are locked, not low-frequency idle",
        "interpretation": "S is an opportunity proxy, not measured energy savings or a confidence interval",
        "power_max_gap_seconds": MAX_GAP_SECONDS,
        "buffer_note": "B is each episode's sampled buffer_summary_actions converted by control_freq_hz; medians are per-episode summaries, not a pooled global median",
        "episodes": episodes, "by_q": q_summaries,
    }
    _write(run_dir / "opportunity.json", result)
    (run_dir / "opportunity.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _write(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".opportunity-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fmt(value):
    return "—" if value is None else "{:.4f}".format(value)


def _pct(value):
    return "—" if value is None else "{:.1f}%".format(value * 100.0)


def _markdown(result):
    rows = [
        "# M1.2 推理机会初步分析", "",
        "- run 状态：`{}`（episodes {} / {}）".format(
            result["status"], result["observed_episodes"], result["expected_episodes"]),
        "- 功率轨：VDD_IN 模块输入，包含渲染和控制负载。idle 是同次运行内无模型推理的时间补集；设备锁频，不能称作低频 idle。",
        "- `S=(1-d)(1-r)` 是机会估计，不是实测节能收益或置信区间。能量单位为 J；积分窗口缺测时完整能量、均功率及其派生量留空。",
        "- B 来自每 episode 的 buffer 摘要乘以控制频率倒数；中位数是 episode 内采样摘要，不合并为全局中位数。", "",
        "## 每 episode", "",
        "| task | q | 源状态 | 分析 | d | busy/idle 覆盖率 | P_on W | P_idle W | r | S | busy/idle J（完整） | B min/median/max s |",
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for e in result["episodes"]:
        b, i, buf = e["busy_power"] or {}, e["idle_power"] or {}, e["buffer_summary_seconds"]
        rows.append("| {} `{}` | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {}/{} | {}/{}/{} |".format(
            e["task_id"], e["task_name"] or "", e["q"], e["source_status"], e["status"],
            _fmt(e["d"]), _pct(b.get("coverage_fraction")), _pct(i.get("coverage_fraction")),
            _fmt(e["P_on_w"]), _fmt(e["P_idle_w"]), _fmt(e["r"]), _fmt(e["S"]),
            _fmt(b.get("total_energy_j")), _fmt(i.get("total_energy_j")),
            _fmt(buf["min"]), _fmt(buf["median"]), _fmt(buf["max"])))
    rows += ["", "## 按 q 汇总", "",
             "| q | 状态 | episodes / expected | d（partial 时 observed） | busy/idle 覆盖率 | P_on W | P_idle W | r | S | busy/idle 覆盖积分 J | busy/idle 完整能量 J |",
             "|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|"]
    for q in result["by_q"]:
        rows.append("| {} | {} | {} / {} | {} | {}/{} | {} | {} | {} | {} | {}/{} | {}/{} |".format(
            q["q"], q["status"], q["episodes"], q["expected_episodes"],
            _fmt(q["d"] if q["d"] is not None else q["observed_d"]),
            _pct(q["observed_busy_coverage_fraction"]),
            _pct(q["observed_idle_coverage_fraction"]),
            _fmt(q["P_on_w"]), _fmt(q["P_idle_w"]), _fmt(q["r"]), _fmt(q["S"]),
            _fmt(q["busy_covered_energy_j"]), _fmt(q["idle_covered_energy_j"]),
            _fmt(q["busy_energy_j"]), _fmt(q["idle_energy_j"])))
    return "\n".join(rows) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="含 summary.json 的显式 run 目录")
    args = parser.parse_args(argv)
    try:
        result = analyze_run(args.run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print("{} ({})".format(args.run_dir, result["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
