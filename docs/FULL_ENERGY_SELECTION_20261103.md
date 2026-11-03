# Full 按总能量选频

2026-11-03。仅改 Full 的选频依据，保留原中频优先方案。Fast关闭，相机采集、模型、RTC、关键动作规则、25ms余量与0.2s降频门槛均不改。

## 实现

原方案先看中频能否赶上，能则中频，否则高频。新方案在原本要准备 Full 时比较两个合法候选：中频Full＋之后的Continue，高频Full＋之后的Continue。

先沿用原P95、关键截止、RTC前缀和首次刷新限制排除赶不上的配置，再比较模块输入总能量。服务均值仅用于估算成本，不替换原P95时限。

- Full成本从准备设备开始，包含恢复/升频、感知、专家和提交，不再另外加一遍升频能量。
- Full结束后的降频窗口和Continue单列。若预计等待达到原0.2s门槛，按原规则估算降到306MHz的成本；短等待按计算频率保持计价。
- 保持功率按刚结束的Full频率以及0–0.2s、0.2–1s、1s以上的实测响应分段积分，保留推理后功率回落过程。
- 两候选使用相同准备触发和能量账期。账期截止取当前已有动作的缓冲边界；关键动作截止单独用于服务准入，不能混为一个时刻。账期用于估价，Full产生新动作后仍照常接续和重新规划。
- 某条入口路径缺数据时不填0；同路径缺对应等待历史时，比较已有历史各自配对的时间和能量，取较贵的整窗估计。两个配置均已赶不上时，沿用原紧急处理。

本地选频计算约0.10ms/次，只在Full准备触发时运行。能量是局部预测，新增动作可能改变下一次刷新；最终收益以实际整任务积分判断。

## 数据与离线检查

复用34条已审核的原有运行，含186次常规Full，没有重新测试q30。提取器新增降频命令至实际读回窗口的绝对能量，使用多次观测的平均值；这段替代同时间的保持成本，避免重复收费。提取器拒绝混入Fast请求。

[成本表](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T081752.553801Z-lean-cost-refresh-c7546b12/cost_table.json>) · [来源运行与分账](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T081752.553801Z-lean-cost-refresh-c7546b12/episode_rows.csv>)。

已有103次三频Full准备决策回放：原76次中频、27次高频；新75次中频、28次高频，只有1次改档。19次原本已不能满足截止的紧急情况保留原处理；其余84次均有可用成本。改变的一次是Object3第54步，由中频改为高频。回放仅说明选择结果，不代表新闭环成功率或实测节能。

[离线回放](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T082759.118212Z-full-energy-buffer-replay-65ee1167/summary.json>) · [回放脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T082759.118212Z-full-energy-buffer-replay-65ee1167/replay.py>)。

75项相关检查通过：同窗计价、历史配对、恢复不重复收费、保持分段、截止限制、缺失成本回退、原准备触发保留、独立记录审查与原三频回归。仅做Object3原/新各一次的常驻配对，其他任务与q30不重测。

## 闭环小测结果

|Object3，同初态|原中频优先|按总能量选择|
|---|---:|---:|
|结果|160步成功|158步成功|
|任务时间|9.773s|9.554s|
|任务＋在途推理能量|131.077J|131.036J|
|Full刷新/其中中频|8/5|8/4|
|Full服务能量|87.808J|86.724J|
|Continue能量|43.269J|44.312J|

有效配对共加载一次模型、一次专家。初始物理状态、图像、state、噪声及首段动作完全一致；982/1008项独立记录检查、实际频率与恢复检查通过，两组零断供。第54步确实由中频改为高频；其他时刻仍采用原频率选择。

主能量仅少0.041J（0.03%），基本相同；本轮任务时间少2.24%。Full服务少1.084J，被Continue多出的1.043J抵消。3秒公共低频测量尾段单列，含尾段能量152.581→154.241J，未计入主结果。单次小测不能认定稳定加速或节能，原中频优先继续作为默认，新选择器保留独立开关。

[配对CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T083415.098568Z-acceptance-m3-66402c57/full_frequency_pairs.csv>) · [分账和逐次选频](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T083415.098568Z-acceptance-m3-66402c57/postrun_review.json>) · [批次记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T083415.098568Z-acceptance-m3-66402c57/summary.json>)。

首轮在原组成功160步后因既有审查边界错误中止，原invalid记录保留。原因是区间判断在103步作出、频率确认后104步提交，审查错误地拿提交步检查104步结束的区间。已改为按原决定步核对来源，控制与推理规则未变；原记录重新审查981项通过。[原批及复核](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T082918.076320Z-acceptance-m3-3cc5bdd6/reaudit_after_boundary_fix.json>)。随后上述有效配对复用了通过的环境缓存；本次工作累计两次模型加载、三条实际任务记录，不把首轮漏记或冒充全程只加载一次。

## 使用

新开关 `--full-energy`，不加则保持原中频优先。配对仍使用 `--full-frequency-pair`，两组为 `full_three_medium` 和 `full_three_energy`，都使用原25ms启动点。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency --full-energy \
  --full-frequency-pair --start-floor-ms 25 --task-indices 4 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

代码：[成本与选择](../perceptbudget/full_energy.py)、[原三频调度接入](../perceptbudget/three_frequency.py)、[独立审查](../scripts/audit_rtc_system.py)。修改前保存于 `archive/before-full-energy-selection-20261103`。
