# M3额外时间余量初筛

入口：`/home/wyk/venvs/lutvla/bin/python -u scripts/run_m3.py --mode margin-scan`

Spatial/Object/Goal各task1，seed7/init0，每任务比较50、25、0ms，共9回合。
只加载一次模型，三个任务轮换余量顺序。完整P95服务耗时仍按原方式在线更新；
只改变额外余量，机制一、joint调频、H50/N5/BF16/state/RTC均保留。
该入口独立于正式四组比较，不修改默认50ms；0ms不表示取消已有RTC提交约束或看门狗。

结果在任务产出/perceptbudget/M3/<编号>-margin-scan/下：
margin_summary.json记录每回合任务结果、真实时间、总能量、推理次数、刷新迟到次数与最大时长、
动作断供步数、频率/功率检查，以及首次预测配对检查。
刷新迟到与动作断供分别统计：错过关键事件截止时间不一定已经耗尽动作。

这是候选筛选。先看初始预测是否一致、任务是否成功、是否迟到/断供，再比较能量。
选定候选后扩展到跨任务较长运行和独立重复，最终才给出本阶段采用的eta；
9回合零断供不能直接声称完成长运行校准或给出稳定的P99模型。

## 50/25ms扩展运行

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m3_margin_long.py
```

两轮独立运行，各Spatial/Object/Goal task1、3、6 x 两档=18回合，总36回合。
每轮一次模型加载，轮内不逐任务重载；两轮独立加载用于观察跨运行波动。
第二轮反转任务顺序，任务内轮换余量顺序。seed7/init0和所有质量失败均保留。
各轮分别保存margin_summary.json、首预测配对审查、完整请求与功率记录。
额外记录服务预算超限次数，并区分决定提交时已来不及的截止时间。
两轮日志是运行重复，同一任务的重复回合不作为不同任务用于置信区间。
会话索引session.json引用各轮目录；出错立即停止，不自动重试或覆盖已完成记录。
默认余量仍50ms，审核结果后再决定是否调整。
