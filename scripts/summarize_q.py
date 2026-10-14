#!/usr/bin/env python3
"""Summarize an eval_q run directory without changing its summary.json."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = ("success", "failure", "timeout", "invalid")


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _count(value):
    value = _num(value)
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _episode(ep, period):
    status = ep.get("status")
    inference = ep.get("inference") or {}
    power = ep.get("power") or {}
    coverage = _num(power.get("coverage_fraction"))
    energy_mj = _num(power.get("total_energy_mj"))
    energy_complete = (
        energy_mj is not None and coverage is not None and coverage >= 1.0 - 1e-9
    )
    raw_buffer = inference.get("buffer_summary_actions") or {}
    buffer = {"count": raw_buffer.get("count")}
    for name in ("min", "median", "max"):
        actions = _num(raw_buffer.get(name))
        buffer[name + "_s"] = (
            actions * period if actions is not None and period is not None else None
        )
    underflow = inference.get("underflow_steps")
    dropped = _count(inference.get("dropped_actions"))
    expired = _count(inference.get("expired_actions"))
    overwritten = _count(inference.get("overwritten_unused_actions"))
    busy = _num(inference.get("inference_busy_seconds_within_task"))
    real = _num(ep.get("real_task_seconds"))
    return {
        "task_id": ep.get("task_id"), "q": ep.get("q"), "status": status,
        "mode": ep.get("mode"), "execution_mode": ep.get("execution_mode"),
        # eval_q records control time for every terminal status; report it only on success.
        "simulated_control_seconds": (
            _num(ep.get("simulated_control_seconds")) if status == "success" else None
        ),
        "real_task_seconds": real,
        "energy_j": energy_mj / 1000.0 if energy_complete else None,
        "energy_coverage_fraction": coverage,
        "energy_complete": energy_complete,
        "underflow_count": len(underflow) if isinstance(underflow, list) else None,
        "dropped_actions": dropped,
        "expired_actions": expired,
        "overwritten_unused_actions": overwritten,
        "busy_seconds": busy,
        "buffer_summary_seconds": buffer if raw_buffer else None,
    }


def _q_summary(q, rows, task_ids):
    counts = Counter(row["status"] for row in rows)
    successes = [row for row in rows if row["status"] == "success"]
    control_times = [row["simulated_control_seconds"] for row in successes
                     if row["simulated_control_seconds"] is not None]
    success_real = [row["real_task_seconds"] for row in successes
                    if row["real_task_seconds"] is not None]
    attempt_real = [row["real_task_seconds"] for row in rows
                    if row["real_task_seconds"] is not None]
    energy = [row["energy_j"] for row in rows if row["energy_j"] is not None]
    underflows = [row["underflow_count"] for row in rows if row["underflow_count"] is not None]
    dropped = [row["dropped_actions"] for row in rows if row["dropped_actions"] is not None]
    expired = [row["expired_actions"] for row in rows if row["expired_actions"] is not None]
    overwritten = [
        row["overwritten_unused_actions"] for row in rows
        if row["overwritten_unused_actions"] is not None
    ]
    busy_rows = [
        row for row in rows if row["busy_seconds"] is not None
        and row["real_task_seconds"] is not None and row["real_task_seconds"] > 0
    ]
    busy_window = sum(row["real_task_seconds"] for row in busy_rows)
    energy_sum = sum(energy) if energy else None
    ids = [row["task_id"] for row in rows]
    grid_complete = (
        bool(task_ids) and len(rows) == len(task_ids)
        and len(set(ids)) == len(task_ids) and set(ids) == set(task_ids)
        and all(row["status"] in STATUSES for row in rows)
    )
    energy_complete = bool(rows) and len(energy) == len(rows)
    return {
        "q": q, "attempt_count": len(rows),
        "q_grid_complete": grid_complete,
        "success_count": counts["success"], "failure_count": counts["failure"],
        "timeout_count": counts["timeout"], "invalid_count": counts["invalid"],
        "success_rate": counts["success"] / len(rows) if rows else None,
        "pending_count": len(rows) - sum(counts[s] for s in STATUSES),
        "successful_control_time_mean_s": (
            _mean(control_times) if successes and len(control_times) == len(successes) else None
        ),
        "successful_real_time_mean_s": (
            _mean(success_real) if successes and len(success_real) == len(successes) else None
        ),
        "all_attempts_real_time_mean_s": (
            _mean(attempt_real) if rows and len(attempt_real) == len(rows) else None
        ),
        "all_attempts_real_time_sample_count": len(attempt_real),
        "reported_attempt_energy_sum_j": energy_sum,
        "energy_complete_attempt_count": len(energy),
        "energy_incomplete_attempt_count": len(rows) - len(energy),
        "energy_sum_complete": energy_complete,
        "energy_per_success_j": (
            energy_sum / counts["success"]
            if grid_complete and energy_complete and counts["success"] and energy_sum is not None
            else None
        ),
        "underflow_step_count": (
            sum(underflows) if rows and len(underflows) == len(rows) else None
        ),
        "underflow_reported_episode_count": len(underflows),
        "dropped_actions_total": (
            sum(dropped) if rows and len(dropped) == len(rows) else None
        ),
        "dropped_actions_reported_episode_count": len(dropped),
        "expired_actions_total": (
            sum(expired) if rows and len(expired) == len(rows) else None
        ),
        "expired_actions_reported_episode_count": len(expired),
        "overwritten_unused_actions_total": (
            sum(overwritten) if rows and len(overwritten) == len(rows) else None
        ),
        "overwritten_unused_actions_reported_episode_count": len(overwritten),
        "busy_duty_fraction_weighted": (
            sum(row["busy_seconds"] for row in busy_rows) / busy_window
            if busy_window else None
        ),
    }


def _paired_metric(q5, qrows, field):
    pairs = [
        (q5[task][field], qrows[task][field]) for task in sorted(set(q5) & set(qrows))
        if q5[task][field] is not None and qrows[task][field] is not None
    ]
    base, candidate = _mean([p[0] for p in pairs]), _mean([p[1] for p in pairs])
    return {
        "paired_count": len(pairs), "q5_mean_s": base, "q_mean_s": candidate,
        "delta_mean_s": candidate - base if base is not None and candidate is not None else None,
    }


def summarize(run_dir):
    run_dir = Path(run_dir)
    source_path = run_dir / "summary.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    episodes = source["episodes"]
    frequency = _num(source["control_freq_hz"])
    period = 1.0 / frequency if frequency and frequency > 0 else None
    rows = [_episode(ep, period) for ep in episodes]
    def run_mode(key):
        value = source.get(key)
        if value is not None:
            return value
        values = [ep.get(key) for ep in episodes if ep.get(key) is not None]
        return values[0] if values and all(item == values[0] for item in values) else None

    mode = run_mode("mode")
    execution_mode = run_mode("execution_mode")
    q_values, task_ids = source["q_values"], source["task_ids"]
    task_suites = {task["task_id"]: task.get("suite", "libero_spatial")
                   for task in source.get("tasks", [])}
    for row, episode in zip(rows, episodes):
        row["suite"] = episode.get("suite", task_suites.get(row["task_id"], "libero_spatial"))
    suite_ids = defaultdict(list)
    for task_id in task_ids:
        suite_ids[task_suites.get(task_id, "libero_spatial")].append(task_id)
    by_q = {q: [row for row in rows if row["q"] == q] for q in q_values}
    q_summaries = [_q_summary(q, by_q[q], task_ids) for q in q_values]

    expected = source["expected_episodes"]
    pairs = Counter((row["task_id"], row["q"]) for row in rows)
    expected_pairs = {(task, q) for task in task_ids for q in q_values}
    grid_complete = (
        len(rows) == expected and len(expected_pairs) == expected
        and all(pairs[pair] == 1 for pair in expected_pairs)
    )
    all_terminal = all(row["status"] in STATUSES for row in rows)
    complete = source["status"] == "completed" and grid_complete and all_terminal

    comparisons = []
    if 5 in by_q:
        q5 = {
            row["task_id"]: row for row in by_q[5] if row["status"] == "success"
        }
        for q in q_values:
            candidate = {
                row["task_id"]: row for row in by_q[q] if row["status"] == "success"
            }
            common = sorted(set(q5) & set(candidate))
            comparisons.append({
                "q": q, "common_success_task_ids": common,
                "common_success_count": len(common),
                "control_time": _paired_metric(q5, candidate, "simulated_control_seconds"),
                "real_time": _paired_metric(q5, candidate, "real_task_seconds"),
            })

    return {
        "source_status": source["status"],
        "status": "completed" if complete else "partial",
        "mode": mode, "execution_mode": execution_mode,
        "expected_episodes": expected, "observed_episodes": len(rows),
        "control_freq_hz": frequency, "control_period_s": period,
        "q_values": q_values, "q_summaries": q_summaries,
        "suite_summaries": [
            {"suite": suite, "task_count": len(ids), "q_summaries": [
                _q_summary(q, [row for row in by_q[q] if row["task_id"] in ids], ids)
                for q in q_values]}
            for suite, ids in suite_ids.items()],
        "q5_paired_comparisons": comparisons, "episodes": rows,
    }


def _fmt(value):
    return "—" if value is None else "{:.3f}".format(value)


def _fmt_count(value, reported, total):
    if value is None:
        return "— ({}/{}有记录)".format(reported, total)
    return str(value)


def render_markdown(report):
    suite_q = {
        suite["suite"]: {item["q"]: item for item in suite["q_summaries"]}
        for suite in report.get("suite_summaries", [])
    }
    suite_names = (
        ("libero_spatial", "Spatial"),
        ("libero_object", "Object"),
        ("libero_goal", "Goal"),
    )
    rows = [
        "# q 评测汇总", "",
        "- 状态：{}；源状态：`{}`".format(
            "已完成" if report["status"] == "completed" else "partial（未齐，非最终结果）",
            report["source_status"],
        ),
        "- Episodes：{} / {}；控制频率：{} Hz（周期 {} s）".format(
            report["observed_episodes"], report["expected_episodes"],
            _fmt(report["control_freq_hz"]), _fmt(report["control_period_s"]),
        ),
        "- mode：`{}`；execution_mode：`{}`{}".format(
            report.get("mode") if report.get("mode") is not None else "未知",
            report.get("execution_mode") if report.get("execution_mode") is not None else "未知",
            "；同步质量测试，能耗不代表异步节能"
            if report.get("execution_mode") == "synchronous_quality" else "",
        ),
        "", "## {}".format(
            "最终总表" if report["status"] == "completed" else "阶段汇总（非最终）"
        ), "",
        "集合列为成功数（实际次数/目标任务数）；成功时间为成功任务仿真时间均值/真实时间均值；能量仅在任务格与能耗覆盖均完整时给出。",
        "",
        "| q | Spatial 成功（实际/目标） | Object 成功（实际/目标） | Goal 成功（实际/目标） | 总成功/实际 | 成功率 | 成功时间：仿真/真实 (s) | J/成功 | underflow 步/expired 动作 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report["q_summaries"]:
        suite_cells = []
        for suite_key, _label in suite_names:
            item = suite_q.get(suite_key, {}).get(q["q"], {})
            target = next((entry["task_count"] for entry in report.get("suite_summaries", [])
                           if entry["suite"] == suite_key), 0)
            suite_cells.append("{} ({}/{})".format(
                item.get("success_count", 0), item.get("attempt_count", 0), target,
            ))
        success_rate = (
            "{:.1f}%".format(q["success_rate"] * 100)
            if q["success_rate"] is not None else "—"
        )
        action_counts = "{} / {}".format(
            _fmt_count(q["underflow_step_count"],
                       q["underflow_reported_episode_count"], q["attempt_count"]),
            _fmt_count(q["expired_actions_total"],
                       q["expired_actions_reported_episode_count"], q["attempt_count"]),
        )
        rows.append(
            "| {} | {} | {} | {} | {}/{} | {} | {} / {} | {} | {} |".format(
                q["q"], *suite_cells, q["success_count"], q["attempt_count"],
                success_rate, _fmt(q["successful_control_time_mean_s"]),
                _fmt(q["successful_real_time_mean_s"]),
                _fmt(q["energy_per_success_j"]), action_counts,
            )
        )
    rows.extend(["", "## 分集合统计", "",
                 "| 集合 | q | 已测 / 任务数 | 成功 | 超时 | invalid | 每成功任务能量 J |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
    for suite in report.get("suite_summaries", []):
        for item in suite["q_summaries"]:
            rows.append("| {} | {} | {} / {} | {} | {} | {} | {} |".format(
                suite["suite"], item["q"], item["attempt_count"], suite["task_count"],
                item["success_count"], item["timeout_count"], item["invalid_count"],
                _fmt(item["energy_per_success_j"])))
    rows.extend([
        "", "## 能耗与动作刷新统计", "",
        "| q | 全尝试能耗小计 (J) | 每成功任务能耗 (J) | 完整覆盖数 / 次数 | underflow 步数 | expired actions | overwritten unused actions | dropped actions | busy 占空比（耗时加权） |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for q in report["q_summaries"]:
        energy = _fmt(q["reported_attempt_energy_sum_j"])
        per_success = _fmt(q["energy_per_success_j"])
        coverage = "{}/{}{}".format(
            q["energy_complete_attempt_count"], q["attempt_count"],
            "（完整）" if q["energy_sum_complete"] else "（未完整）",
        )
        if not q["energy_sum_complete"]:
            energy += "（小计）"
            if q["energy_per_success_j"] is not None:
                per_success += "（未完整）"
        duty = q["busy_duty_fraction_weighted"]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                q["q"], energy, per_success, coverage,
                _fmt_count(q["underflow_step_count"],
                           q["underflow_reported_episode_count"], q["attempt_count"]),
                _fmt_count(q["expired_actions_total"],
                           q["expired_actions_reported_episode_count"], q["attempt_count"]),
                _fmt_count(q["overwritten_unused_actions_total"],
                           q["overwritten_unused_actions_reported_episode_count"], q["attempt_count"]),
                _fmt_count(q["dropped_actions_total"],
                           q["dropped_actions_reported_episode_count"], q["attempt_count"]),
                "—" if duty is None else _fmt(duty * 100) + "%",
            )
        )
    rows.extend([
        "", "## 与 q=5 的共同成功任务比较", "",
        "| q | 共同成功 task_id | 控制时间：q5 / q / Δ (s) | 真实时间：q5 / q / Δ (s) |",
        "|---:|---|---:|---:|",
    ])
    for pair in report["q5_paired_comparisons"]:
        control, real = pair["control_time"], pair["real_time"]
        rows.append("| {} | {} ({}) | {} / {} / {} | {} / {} / {} |".format(
            pair["q"], pair["common_success_count"],
            ", ".join(map(str, pair["common_success_task_ids"])) or "无",
            _fmt(control["q5_mean_s"]), _fmt(control["q_mean_s"]),
            _fmt(control["delta_mean_s"]), _fmt(real["q5_mean_s"]),
            _fmt(real["q_mean_s"]), _fmt(real["delta_mean_s"]),
        ))
    rows.extend([
        "", "## 逐 episode 能耗、动作计数与 buffer", "",
        "buffer min/median/max 按控制频率换算为秒，保留每个 episode 的摘要，不合并各 episode 的 median。",
        "",
        "| task_id | q | 结果 | 能耗 (J) | 覆盖 | underflow 步 | expired actions | overwritten unused actions | buffer count；min / median / max (s) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for ep in report["episodes"]:
        b = ep["buffer_summary_seconds"]
        buffer = "—" if b is None else "{}；{} / {} / {}".format(
            b["count"], _fmt(b["min_s"]), _fmt(b["median_s"]), _fmt(b["max_s"])
        )
        cov = ep["energy_coverage_fraction"]
        coverage = "未记录" if cov is None else _fmt(cov * 100) + "%"
        if cov is not None and not ep["energy_complete"]:
            coverage += "（未完整）"
        rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            ep["task_id"], ep["q"], ep["status"], _fmt(ep["energy_j"]), coverage,
            "—" if ep["underflow_count"] is None else ep["underflow_count"],
            "—" if ep["expired_actions"] is None else ep["expired_actions"],
            "—" if ep["overwritten_unused_actions"] is None
            else ep["overwritten_unused_actions"], buffer,
        ))
    rows.extend([
        "", "口径：控制时间只统计成功任务；q5 对照只取共同成功 task_id。",
        "能耗仅使用 power.total_energy_mj 转 J；覆盖不全不外推，表中总量为完整覆盖尝试小计。每成功能耗仅在本 q 任务格和能耗覆盖均完整时计算；零成功为 —。",
        "同步质量模式的能耗和真实时间仅描述同步运行，不代表异步闭环或异步节能。缺失的旧记录字段显示为未知（—），不按零计。",
        "P_onidle 尚未独立测量，本汇总不计算节能机会或作节能结论。",
    ])
    return "\n".join(rows) + "\n"


def write_report(run_dir):
    run_dir = Path(run_dir)
    report = summarize(run_dir)
    json_path, md_path = run_dir / "q_comparison.json", run_dir / "q_comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="包含 summary.json 的明确 run 目录")
    args = parser.parse_args(argv)
    try:
        json_path, md_path = write_report(args.run_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, "汇总失败：{}\n".format(exc))
    print("已写入 {} 和 {}".format(json_path, md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
