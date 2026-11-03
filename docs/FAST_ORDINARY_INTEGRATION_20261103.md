# 普通阶段接入 Fast

2026-11-03。保留当前三频 Full＋Continue，只在普通阶段允许单次 Fast。未改 FlashRT、checkpoint、RTC 采样算法、N5 或 BF16。

## 怎么选择

1. 原机制一寻找夹爪切换、减速、转向；原 Full 预算决定刷新时间，中频能赶上则用中频，否则高频。
2. 当前处于关键操作阶段，或本次刷新由关键动作触发，使用 Full。只有普通阶段且上次不是 Fast，才比较候选。
3. 两个候选使用相同启动时间、相同计算频率；比较本次计算与同一截止时间前 Continue 的能量。公共频率准备成本相同，在差值中抵消。Continue 单列，不记到 Fast 推理上。
4. Fast 更省且其延迟能放入原 Full 窗口，才替换本次 Full；下一次必须 Full。频率确认后再次检查当前阶段，避免准备期间已经进入关键阶段。
5. 沿用当前异步 RTC、按实际步数裁剪、关键动作等待、阶段去重及首次刷新约束。Fast 的服务样本不进入 Full 的耗时历史，延迟历史按模式和计算频率分别维护。

Full：双图512视觉 token＋state。Fast：双图各160，共320视觉 token，合并每路最冗余的32个2×2块，省略 state，无新训练。候选来源见[双路选择压缩小测](FAST_DUAL_SELECTIVE_20261103.md)。

## 新成本数据

同一模型和三份保存输入，每条件6次；固定输入成本仅用于候选排序，闭环收益用完整任务的实际积分能量判断。

|计算配置|Full平均耗时|Fast平均耗时|Full请求能量|Fast请求能量|
|---|---:|---:|---:|---:|
|高频1173/3199|474ms|339ms|11.276J|7.932J|
|中频816/2133|677ms|488ms|10.539J|7.402J|

计时、能量均从频率确认后的固定输入请求开始；不含之前的调频。GPU/EMC单位为MHz，CPU固定1497.6MHz。另测同频率保持功率，高频12.347W、中频8.658W，用于给同窗剩余等待计价。这是短测的保持功率，不替代实际 Continue 的逐段功率。

[原始成本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T073037.005184Z-dual-selective-fast-b4a3c6ec/ordinary_fast_costs.json>) · [原始功率](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T073037.005184Z-dual-selective-fast-b4a3c6ec/clock_cost_power.json>)。所有条件频率检查通过，已恢复原频率。

## 运行与验证

入口沿用 `scripts/run_m4.py`，增加 `--ordinary-fast`；加 `--ordinary-fast-pair` 即与原三频 Full＋Continue 常驻配对。原入口不加此参数时保持原流程。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency --start-floor-ms 25 \
  --ordinary-fast --ordinary-fast-pair --task-indices 0 3 7 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

同批 Spatial0、Object0、Goal3，各原流程/候选一次，共6回合、1次模型加载；固定 seed7/init0，比较实际成功、任务耗时、任务＋在途推理能量，3秒公共低频尾段单列。检查初态、首段动作、关键阶段选择、Full时间表、独立延迟历史、频率、动作来源及能量分账。

## 本轮结果

|任务|原流程：结果/步数|加入Fast：结果/步数|原流程能量|加入Fast能量|实际Fast次数|
|---|---|---|---:|---:|---:|
|Spatial0|成功/75|成功/76|69.47J|67.84J|1|
|Object0|成功/145|超时/280|110.52J|218.62J|3|
|Goal3|成功/174|成功/172|127.80J|128.22J|2|

上表为任务＋在途推理能量，排除公共低频尾段。Spatial省2.35%，Goal多0.33%；两个共同成功任务合计197.27→196.06J，省0.61%。原流程3/3成功、候选2/3；不能认定整体改善。

六回合共加载一次模型、一次专家；三对初始物理状态、双图、state、噪声、首段动作完全一致，6,065项独立记录检查、频率/恢复检查通过，全部零断供。Fast实际在29/59/73/75/239等普通位置执行，后次均为Full，没有把纯等待算作Fast。

Object0首次替换在29步，当时当前预测没有未覆盖的关键事件；随后候选在40步、原流程在43步启动关键位置Full。输入压缩和后续预测时序已经不同，失败不能只归结为单个公式。Goal候选总刷新7→8次，抵消了部分单次Fast收益。当前“预测中没有关键事件”不足以保证Fast不会影响后续抓取，保留实验开关，原Full＋Continue继续作为基线。

准备期间可能跨一个控制步，因此增加了提交前的当前阶段复核；六回合保存的23次非首轮请求按该复核离线重算，模式均未改变，另有对应单测。Full刷新时间、计算频率与原窗口均不因该复核变化。

[配对结果CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T073603.551457Z-acceptance-m3-11d83a4d/full_frequency_pairs.csv>) · [逐例审查与原始运行链接](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T073603.551457Z-acceptance-m3-11d83a4d/postrun_review.json>) · [批次记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T073603.551457Z-acceptance-m3-11d83a4d/summary.json>)。

代码：[模式选择](../perceptbudget/ordinary_fast.py)、[原三频调度后的接入点](../perceptbudget/three_frequency.py)、[独立记录审查](../scripts/audit_rtc_system.py)。修改前保存于 Git 分支 `archive/before-three-frequency-ordinary-fast-20261103`。
