# Full推理期间跳过无用成像

每次模型请求使用启动时取得的一组外部＋腕部图像。旧精简模式只在没有推理在途时关闭相机，推理期间仍会逐步渲染；新模式在所有动作执行步暂停图像传感器，下一次请求前补采最新双图。物理仿真、本体state、接触记录和动作执行保持原流程，不改变FlashRT、模型、RTC、调度或频率选择规则。

项目入口新增`--request-only-images`。它依赖原精简采集，适用于M3 Full＋Continue和q30；不传时保留现有行为。旧图像不会冒充新观测：暂停相机的控制步不返回image字段，提交前使用既有camera-only强制采集，保持该步state和物理状态。

`--request-capture-pair`在同一常驻模型中比较`inflight_images`（当前版本）与`request_images`（新模式），两组均保留三档、中频Full优先、50ms空档门槛、25ms启动点、关键等待/阶段刷新及原RTC。任务时间、任务＋在途推理能量、共同终态3秒尾段分别统计。两组首图/state/噪声/raw动作须精确配对；后续异步时序随执行速度变化，按实际数据报告。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency \
  --idle-min-ms 50 --start-floor-ms 25 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh \
  --task-indices 0 --request-only-images --request-capture-pair
```

只测Spatial task0，两回合、一次模型加载。图像仍为完整双视角，state开启，H50、N5、BF16、seed7/init0不变；两组每次请求均保存画面，附真实控制步索引，不作为逐步视频。额外记录推理期间跳过的成像步数。

65项相关CPU检查通过，包括相机恢复、物理/state继续执行、补采最新双图、相同调度配置的配对、能量尾段划分、实际子进程参数转发。正式配置与既有12任务结果保留。

## Spatial0同任务对照结果

运行`M4/20261104T004414.748951Z-acceptance-m3-e17209f7`，代码`106a5d8`。复用已通过的单任务MuJoCo环境缓存，准备仅1.31秒；一次模型/专家加载，随后常驻完成两个回合。两组均四次Full，其中两次中频；没有Fast。

|指标|当前模式：推理期间成像|新模式：仅请求时成像|
|---|---:|---:|
|结果/步数|成功/75|成功/76|
|任务完成时间|5.758s|5.634s|
|任务＋在途推理能量|69.438J|67.049J|
|Full次数|4|4|
|跳过成像的控制步数|50|76|
|其中推理在途时跳过成像|0|29|
|保存的请求画面数|4|4|
|补采双图次数|3|3|
|补采中位耗时|6.47ms|6.51ms|
|实际控制频率|15.14Hz|15.68Hz|

新模式任务＋在途推理总能量降低3.44%，任务时间缩短2.15%。两组均零断供，初始物理/双图/state/噪声/首raw动作精确配对；每次补采不推进物理、不重新采样state，请求state与对应执行步记录逐值相同。图像索引、独立系统审查、计算频率及恢复均通过，六个能量窗口从原始功率重新积分差0J。结束读回GPU1173/EMC3199/CPU1497.6MHz。

这是单任务配对结果。减少渲染同时改变了实际执行速度和后续异步观测时机，不能把3.44%全部解释成独立的相机功耗下降，也不直接外推十二任务节能率。新模式通过`--request-only-images`启用；当前默认和原12任务基线保留，可继续配对扩测。

[配对总表](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T004414.748951Z-acceptance-m3-e17209f7/request_capture_pairs.csv>) · [逐项复核](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T004414.748951Z-acceptance-m3-e17209f7/postrun_review.json>) · [复核脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T004414.748951Z-acceptance-m3-e17209f7/review.py>) · [频率恢复读回](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T004414.748951Z-acceptance-m3-e17209f7/clock_after.json>)。
