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
