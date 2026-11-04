# 简化关键刷新小测

目的：用已有q30间隔兜底，检查只保留夹爪开合的提前刷新是否能减少漏判与反复规划。

原组沿用当前机制一；候选仅做两项改变：从每份观测起最多30个控制步启动下一次预测，预测动作中的夹爪开合可提前启动。计数包括推理期间已执行的动作，不从新结果提交时重新计数。RTC前缀、阶段去重、关键等待、Full中频优先及低频Continue均保持原实现。无Fast、无训练，FlashRT不修改。

选择Spatial2/9、Goal5/6，各两种规则、固定seed7/init0，共8回合，交替组序、共用一次模型加载。仍执行原系统、严格初始配对、时钟及恢复检查；候选另检查每次观测间隔不超过30步、提前事件只来自夹爪开合。能量统一采用任务加在途推理收尾，包含超时回合，3秒终态等待单列。

## 结果

|任务|原规则|夹爪＋30步上限|
|---|---:|---:|
|Spatial2|114步成功|90步成功|
|Spatial9|220步超时|220步超时|
|Goal5|300步超时|300步超时|
|Goal6|82步成功|300步超时|

有效四对中，成功2/4→1/4；包含超时的总能量549.73→697.29J（+26.84%），任务时间45.33→55.23s（+21.83%），Full次数30→41。共同成功仅Spatial2，能量106.88→73.31J、时间8.70→6.36s，不能据此评价总体收益。全部有效回合缓冲耗尽0，独立系统检查9669项及频率/恢复检查通过。

规则确实生效：候选普通阶段通常间隔29步采观测，符合最多30步的上限；原规则普通阶段常见33–36步。29步来自保留的频率准备与启动规则，不是把q重新定义成提交后计数。

**不采用这版简化规则。** Goal6原组在38步根据46步的减速提示启动高频Full；候选删除减速提示，等到53步才再次观测。两组38步前动作和物理状态完全一致，首次执行动作分歧出现在46步，随后候选失败。这说明只看夹爪会漏掉有用信号；本实验同时改变信号及刷新上限，不能据此单独断言30步上限有害，也不把所有失败归为一次原因。

## 配对与归档

实际共跑10回合、两次模型加载，接受8回合。第一批8回合中前三对初态/图像/state/噪声/首动作完全相同；Goal6初始图像不同、首raw最大差0.01506，该对两组均超时，配对拒收。第一批原始状态保留invalid，只补跑Goal6两回合；新配对初始条件和首动作完全一致，原组82步成功、候选超时。有效统计为前三对加新Goal6，不隐藏被拒收记录。

候选代码`f64d250`保存于`archive/gripper-q30-candidate-20261104`，修改前基线为`archive/before-gripper-q30-20261104`。主线撤回本轮运行代码与候选测试，恢复原事件、阶段去重、RTC已知停止前缀修正及原调频策略。没有修改FlashRT或用户周报。

恢复后171项相关检查通过，三个运行源码文件与修改前`b6ca29b`完全一致。

- [有效四对汇总及全部检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/summary.json>)、[逐回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/results.csv>)
- [Goal6首次分歧](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/goal6_timing_review.json>)、[前三对干预前一致性](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/first_request_difference.json>)
- [第一批原始配对检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045317.294800Z-acceptance-m3-a38e9898/paired_bootstrap_audit.json>)、[补跑原始汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045923.828714Z-acceptance-m3-7e0e2ad0/summary.json>)
- [实验入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/pair.py>)、[复核入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/review.py>)：需候选归档分支；主线已撤回该规则。
- [结束频率读数](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T045217.739775Z-gripper-q30-probe-f2f095a5/clock_after.json>)：GPU1173MHz/EMC3199MHz/CPU1497.6MHz。
