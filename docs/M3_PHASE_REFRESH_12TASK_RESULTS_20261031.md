# M3阶段刷新去重：12任务结果（2026-10-31）

配置：H50/N5/BF16/state/RTC；Full＋Continue，无Fast。首次无事件请求上限30步，关键位置等待、阶段内等待去重和事件刷新去重均启用。四suite各task0/3/6，seed7、init0。模型加载一次，12条完成、系统审计全部通过、无断供。

| 方案 | 成功 | 总任务耗时(s) | 总任务能量(J) |
|---|---:|---:|---:|
| 历史原M3 | 8/12 | 152.58 | 2187.45 |
| 历史固定q30 | 10/12 | 126.11 | 1859.42 |
| 当前M3实验 | 11/12 | 117.88 | 1576.86 |

总能耗和时间包含超时任务，成功差异会显著改变总量。共同成功的8任务对原M3：能量−4.03%，时间+0.32%；共同成功的10任务对q30：能量−1.60%，时间+5.71%。整任务能量沿用既定计量范围，包括收尾；时间列为task_seconds。

| 集合 | task | 结果 | 步数 | 时间(s) | 能量(J) |
|---|---:|---|---:|---:|---:|
| libero_spatial | 0 | success | 76 | 6.01 | 68.97 |
| libero_spatial | 3 | success | 87 | 7.25 | 87.83 |
| libero_spatial | 6 | success | 109 | 8.28 | 107.36 |
| libero_object | 0 | success | 145 | 8.47 | 118.72 |
| libero_object | 3 | success | 125 | 7.60 | 120.01 |
| libero_object | 6 | success | 146 | 8.50 | 119.94 |
| libero_goal | 0 | timeout | 300 | 16.93 | 210.05 |
| libero_goal | 3 | success | 173 | 10.03 | 143.45 |
| libero_goal | 6 | success | 89 | 5.82 | 80.06 |
| libero_10 | 0 | success | 235 | 14.49 | 193.50 |
| libero_10 | 3 | success | 226 | 12.61 | 166.52 |
| libero_10 | 6 | success | 213 | 11.88 | 160.45 |

76次请求，6次等待，累计2.022秒。失败仅Goal0；历史原M3失败的Object3/Goal6/Long0本轮成功。Object3此前相同配置单测曾失败，本轮成功，因此不能把一次成功归因于某一机制；本次多个改动合用，也未分离各自贡献。当前结果支持保留候选、继续检查失败与重复性，不据单轮替换历史基线。

原始批次：/home/wyk/workspace/任务产出/perceptbudget/M4/20261031T063736.148606Z-acceptance-m3-68053278
历史对照：/home/wyk/workspace/任务产出/perceptbudget/M4/20261030T052140.445693Z-m3-completion-plan-1d3cd769/completed_comparison.json
