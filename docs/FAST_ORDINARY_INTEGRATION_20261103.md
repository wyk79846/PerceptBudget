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


## 2026-11-05 原坐标464开发验证与冻结

开发仅用seed7/init0，Long0、Object0、Spatial0/3、Goal0各做Full/Fast配对。原坐标选择器保留图像和文本坐标，并为专家匹配动作坐标；普通缓冲刷新可用Fast，已有夹爪锚点的有限抓放区间仍用Full，禁止连续Fast。Full的刷新计划、选频、H50/N5/BF16/state200及作者RTC采样体不变。

- 原坐标v2为Full4/5、Fast3/5，Long失败；不能把它称为已通过。v3开发对照两组均4/5，Long均成功（225/226步），Goal均超时300步。
- 新v3一模型连续完成10回合，零断供。全部任务及在途推理能量Full601.528J、Fast596.596J，省0.820%；时间49.890/50.682s，增加1.59%。仅小样本，不外推40任务收益。
- 实际8次Fast；中频Full/Fast引导推理中位652.197/528.666ms，接续中位13/11步。Fast在途84步均跳过无用成像；每次新请求补采当前双图。严格配对起点、系统/频率/恢复审核及原始功率复算通过。
- 资源清理在原hard-reset边界完成GL删除并退还CPU空闲页，不改变reset或控制；Long/Object各三次20步重放的物理/state/图像/接触完全相同。模型常驻开发批也未触发RAM保护。仍保留900/768MiB保护线与异常证据。

证据：[开发复算](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T054640.548752Z-fast-phase-dev-r2-e5670d46/review.json>)；[原始父目录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T054715.837417Z-acceptance-m3-a62be153/recording.json>)；[资源释放重放](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T054305.690913Z-reset-release-replay-0a2e1b01/summary.json>)。68相关CPU检查通过，FlashRT未修改。

下一步冻结当前源与成本表后做seed17/init1全40。保留原Full+Continue的35/40和q30的33/40比较，所有失败与在途推理能量均计入；不按验收结果改阈值或筛任务。


## 2026-11-05 验收未通过及后续排查

前述5任务通过不代表全组稳定。no-state464/v3在seed17/init1完成34项，27项成功；同34任务的原Full基线30项成功。所有已完成项及在途推理计入后4120.082J/335.506s，对照3942.054J/317.144s。48次Fast、496个Fast在途控制步均无额外成像。原计划40项，6项因三个准备阶段RAM保护缺失，不能算任务失败或假称已完成40项；已有7个超时，即使补齐也达不到预设35/40。因此停止原样重启，不采用此候选。[封存复算](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T055540.565300Z-energy-fast-frozen-all40-2c782222/rejected_summary.json>)。

内存排查确认旧模拟器数据/模型在重置后已释放，观测数组不持有原生数据。新增可选按需词嵌入读取：BF16词表保持在checkpoint文件映射，只上传当前token行；不改tokenizer、缩放或模型权重。全部257152×2048值与原GPU表逐值一致，Full RTC的50×32输出差0。提前安装后，无仿真固定输入流程GPU保留内存约5932MiB，原约7016MiB；已分配表减少1004.5MiB。长场景常驻仍需验证，不声称所有RAM峰值已解决。该方式只在ordinary Fast候选自动开启；原Full基线保持原配置。[逐值/输出验证](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T065923.567719Z-fast-fixed-input-8ef657ad/summary.json>)。

保留state及重建完整注意力也做了固定输入原型，不能据单帧误差声称闭环成败。完整注意力原型每挡Full/Fast各6样本，结果如下；原作者采样体与本地图在相同KV上差0。

|频率|Full ms|Fast ms|Full J|Fast J|补齐相同时间窗口后的净节能J/次|
|---|---:|---:|---:|---:|---:|
|高频|463.91|440.82|10.865|10.617|-0.0368|
|中频|636.62|615.87|9.940|9.689|0.0708|

快3.3%–5.0%，但净收益不足，不扩大闭环。原型源码保存在`archive/expanded-attention-probes-20261105`（686fd06），从当前代码删除其分支；数据和GPU行路由/动态图检查完整保留。[实测成本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T082939.130085Z-dual-selective-fast-87af879c/common_window_review.json>)。

继续目标：Fast成功率保持稳定并完成seed17全40验收，目前未达到。下一候选先验证只重算变化较大的一个摄像头的SigLIP特征、另一摄像头保留最近Full特征，再用完整VLM和最新state生成新KV；不复用整份旧KV，不连续Fast。尚未实现或验证，不计入已有成果；先用seed7固定输入及配对小测。


## 2026-11-05 单路视觉特征复用小测

只重算变化较大一路SigLIP，另一摄像头保留最近Full特征；两路特征一起经过完整VLM并加入最新state，重算全部KV，RTC不变。未进入生产闭环。Full输出与原记录及Fast后重复Full均差0；作者采样体对照差0。相同图像按单路/双路编码也有BF16算法差异，输出最大差约0.011，不能声称完全等价。

|频率|Full ms|Fast ms|Full J|Fast J|同窗口净节能J/次|
|---|---:|---:|---:|---:|---:|
|高频|464.61|442.01|10.959|10.940|-0.2680|
|中频|640.27|603.15|10.059|9.790|-0.0555|

每模式每挡6个样本，频率读回与恢复通过；速度收益有限，实测同窗口没有净节能，不扩大闭环。缓存第0步视觉特征、在第33步刷新腕部的一帧动作偏差也较大，但单帧不作为成功率结论。代码归档`archive/camera-feature-cache-probe-20261105`（6a6cbe0），主干移除未采用原型。

[原始记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T090015.842587Z-camera-feature-fast-45d9c441/summary.json>) · [能量复算](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T090015.842587Z-camera-feature-fast-45d9c441/common_window_review.json>)。继续保留Full＋Continue基线35/40；Fast稳定验收仍未完成。


## 2026-11-05 保留state与简单交替对照

用户确认先比较全Full、Full/腕部Fast交替、Full/双图温和池化Fast交替。新增显式`--fast-alternate`：首次Full，后续每次刷新严格轮流Fast和Full，不加入阶段准入或能量门槛。刷新和选频仍沿用同一套Full规则，RTC、H50/N5/BF16、完整state不变；Continue仍只执行队列。生产默认与原35/40基线不变，FlashRT源码未修改。76相关检查通过。

腕部单图固定输入成本23f6cd34：高频Full465.32/Fast334.03ms、10.931/7.961J，中640.67/467.15ms、10.036/7.271J；相同时间窗补计Continue后每次净省1.352/1.248J。原五任务配对62f13d36在首个Long0首次推理时触发RAM保护，零完整回合，不报告成功率。此次先从Spatial0开始，按原900/768MiB保护执行，另检查allocator峰值；不能把小场景成功当作长场景内存问题已修复。

保留state的双图464token成本测量f16d841b完成：两摄像头各232token，合并各8/64块。每挡每模式6样本，作者采样体与优化图差0、三个Full与原记录差0。

|频率|Full ms|池化Fast ms|Full J/次|Fast J/次|同时间窗净省J/次|
|---|---:|---:|---:|---:|---:|
|高频|472.19|457.04|11.060|11.255|-0.368|
|中频|641.58|628.50|10.055|10.173|-0.233|

本轮只快2–3%，单次和同窗能量均未下降；仍按约定补同任务交替质量对照，不将它称作有效节能方案。固定输入的夹爪动作也可能变化，不能据此推断闭环成功率。此前27/34为另一版no-state464/v3结果，不能当作本轮stateful交替结果。

[池化成本与作者对照](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T113126.685816Z-dual-selective-fast-f16d841b/summary.json>) · [原始成本表](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T113126.685816Z-dual-selective-fast-f16d841b/ordinary_fast_costs.json>)。

首个Spatial0配对小测已完成，全部seed7/init0、保留state、真实RTC异步。每种Fast分别有本批全Full对照，两个进程的Full绝对能量不同，不能拿一组Full能量给另一组计算收益。

|配对|组别|结果/步数|任务耗时s|任务与在途推理J|请求/其中Fast|
|---|---|---:|---:|---:|---:|
|腕部|全Full|成功/76|5.910|70.146|4/0|
|腕部|Full/腕部交替|成功/79|6.092|77.001|5/2|
|池化|全Full|成功/76|5.622|67.040|4/0|
|池化|Full/双图464交替|成功/79|6.087|69.805|4/2|

本例两种交替均能完成，能量分别增加9.77%和4.12%，尚未证明整任务节能。两对初始物理、双图、state、噪声及原始动作精确，系统/频率/恢复审核均通过，零断供。两种Fast分别9/19个在途控制步全部跳过成像。原腕部父记录因审查器仍检查旧普通阶段准入诊断而标invalid，原始状态与错误不修改；显式交替不依赖该诊断，已修正审查器并重查两个真实回合，独立序列Full/Fast/Full/Fast/Full、state与原RTC索引通过，无需重跑。

[腕部原始数据与更正审查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T113431.307509Z-acceptance-m3-ca08474a/corrected_auditor_review/review.json>) · [池化原始配对](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T114451.857403Z-acceptance-m3-bd1c396b/full_frequency_pairs.json>)。以上只是一任务可行性，不作为总体成功率或长场景内存通过。


## 2026-11-05 双图320token、Fast不带state，直接交替

按用户要求直接闭环小测，Full保留state，Fast双图各160token、不带state，首次Full后严格交替。seed7/init0，三个任务6回合一模型完成，无报错、RAM保护或欠载。

|任务|全Full|Full/Fast交替|全Full J|交替J|
|---|---|---|---:|---:|
|Spatial0|成功76步|成功79步|67.438|65.895|
|Object0|成功136步|超时280步|101.796|204.789|
|Long0|成功247步|超时520步|191.852|371.983|

Full3/3、交替1/3；Fast中频481.3ms、高频352.7ms，相应Full654.5/487.8ms。所有尝试的任务与在途推理能量合计361.086/642.667J；Spatial共同成功省2.29%，不能掩盖另两项超时。实测22次Fast，确认实际state关闭、320token；Full保持state及512token。本批说明速度提升，但质量尚不稳定。未使用或保留前轮新增的packed注意力原型，未修改FlashRT。

[原始配对表](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T122153.255435Z-acceptance-m3-d5fcb279/full_frequency_pairs.json>)。代码7814eb6。
