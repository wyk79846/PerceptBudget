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


## 2026-11-05 腕部256＋外部64、不带state的直接交替

用户指定的原生pool2_agent模式，外部图像做2×2平均池化到64token，腕部256token完整保留；Fast无state，Full保留state。沿用同一三个任务、seed7/init0、RTC、刷新及频率规则，一模型6回合完成，无欠载、报错或内存保护。

|任务|全Full|交替|全Full J|交替J|
|---|---|---|---:|---:|
|Spatial0|成功76步|成功78步|67.309|62.994|
|Object0|成功136步|超时280步|102.152|187.043|
|Long0|成功248步|超时520步|191.511|331.604|

Full3/3、交替1/3，尚未改善这三任务的成功数。Spatial共同成功省6.41%；所有尝试含超时的任务及在途推理360.973/581.641J。Full/Fast中频中位651.7/482.5ms，高频487.7/339.0ms。旧双图320无state成本只作同token形状的时延初值，在配置中明确标注geometry proxy；纯交替不以该成本选模式，表中J均为本批真实任务测量。

[本批原始配对结果](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T123601.494699Z-acceptance-m3-9b7b4e3d/full_frequency_pairs.json>)；代码f7882c7。


## 2026-11-05 池化位置编号检查与Object0小测

原池化的编码器、语言、动作编号前后一致，没有发现相互错位；它使用压缩后的连续编号。新增显式`--ordinary-fast-positions`池化候选：外部2×2块取原左上角token位置，腕部仍为256..511，语言从512起，动作专家同步加192。Full、RTC采样体、H50/N5、state、320token及交替规则不变，不修改FlashRT。只检查了坐标映射并直接运行Object0，没有新增备份分支。

seed7/init0，保留state、外部64＋腕部256、Full/Fast交替：之前连续编号280步超时，本次原编号154步成功、9.507s、0欠载。单例出现改善，尚未扩大测试。

[原始结果](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T134010.066356Z-acceptance-m3-6424ac53/summary.json>)。


## 2026-11-05 原位置编号扩大到12任务

seed7/init0，四套各task0/3/6，同一320token（外部64＋腕部256）、state开启、原位置及动作偏移192、Full/Fast直接交替。12项全部有有效结果，9成功：Spatial3/3、Object3/3、Goal1/3、Long2/3；失败Goal0/6、Long0均超时。Long6补测成功213步；既有Object0成功154步，其余已完成项均未重跑。两次准备阶段内存中断不计作任务失败，五次加载含最早Object0单例，全部原始记录保留。未修改算法或阈值、未重启用户已停止的seed17测试。

[完整12项报告](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T134834.166051Z-acceptance-m3-a0c9c3f7/expanded12_results.md>)。本批为开发样本，不将不同任务组的历史成功率或能量直接当同期对照。

## 2026-11-05 原位置Fast：seed17全40任务

沿用原位置320token（外部64＋腕部256）、state开启、动作位置补偿192及Full/Fast直接交替，seed17/init1，四套各task0–9、每任务一次。全40项完成，33/40成功（82.5%）：Spatial6/10、Object7/10、Goal10/10、Long10/10。超时为Spatial0/1/4/6、Object3/5/6；未剔除失败、未针对该批调整参数。一模型连续完成，零内存保护中断、零断供；166次Full、142次Fast，全部回合已有检查通过，Fast实测请求的state、320token与动作位置偏移均符合配置。

全部尝试包含超时，任务与在途推理总能量5116.780 J，任务执行420.262 s，不含加载和任务准备。非初次推理中位数：高频Full470.9/Fast433.2 ms，中频Full651.2/Fast610.4 ms。该候选在Goal和长任务取得10/10，但Spatial/Object仍有7项超时，暂不替换Full＋Continue基线；不能把本轮与不同成像设置的历史数据直接解释为池化的独立收益。

[全40项结果及逐任务表](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T142840.502775Z-fast-positions-seed17-all40-6f18caf8/results.md>) · [JSON](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T142840.502775Z-fast-positions-seed17-all40-6f18caf8/results.json>) · [原始父记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T142852.290479Z-acceptance-m3-0e288582/recording.json>)。

## 2026-11-06 原位置Fast关闭state：seed17全40任务

按用户要求，只对Fast加`--fast-no-state`，Full仍带state；同seed17/init1、320视觉token（外部64＋腕部256）、原位置编号与动作偏移192、Full/Fast直接交替、H50/N5/BF16及原调度参数。40项一次模型连续完成，无内存保护中断、零断供，全部已有回合检查通过，所有请求的state/位置模式记录符合配置。结果33/40：Spatial9/10、Object7/10、Goal8/10、Long9/10。超时Spatial6、Object0/5/7、Goal3/5、Long8，全部保留。

Fast高频/中频推理中位342.1/482.2ms，相比上一轮带state的433.2/610.4ms均快约21%；Full为477.7/652.7ms。327次请求中150次Fast，平均23.52步刷新；上一轮带state为308次请求、平均24.20步。单次提速没有转成相同比例整任务收益：任务实耗430.566s，全部尝试任务＋在途推理5064.752J，相比纯Full 5253.250J省3.59%（35→33成功）；共同成功30项仅省0.48%。相比带state交替版5116.780J省1.02%，共同成功28项省3.26%，总耗时增加约2.45%。三批为不同进程的固定参数历史对照，无逐任务调参，不把轨迹与成功变化全部归因于state开关。暂不替换Full＋Continue基线。

[完整40项结果与能量对照](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153705.566883Z-fast-pos-nostate-seed17-all40-6658feb2/results.md>) · [JSON与分段能量](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153705.566883Z-fast-pos-nostate-seed17-all40-6658feb2/results.json>) · [原始父记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153753.507758Z-acceptance-m3-9fb292d7/recording.json>)。结束已确认恢复GPU1173/EMC3199/CPU1497.6MHz。

### state长度核对

当前LeRobot checkpoint实际输入8维，配置`max_state_dim=32`；其配套`Pi05PrepareStateTokenizerProcessorStep`先补零到32维，再将全部32维数字写进语言prompt。项目`h50_checkpoint.normalize_state`遵循该格式。200是语言＋state容量上限。用本机实际分词器复算Object0：纯任务文字13token，只编码8维为48token，当前32维为144token；24个补零在量化后成为24个128，贡献96token。这是与配套预处理相符的占位成本，未擅自删除32维输入的尾部。旧Fast无state与新Fast带state的差异，不等于同为带state的Full/Fast差异。

## 2026-11-06 原位置Fast仅在普通阶段刷新

用户同意重试非关键阶段Fast。复用现有feedback_v3准入：普通buffer刷新、未处于夹爪关键区间、上次不是Fast，且同窗能量与时延允许时才使用Fast；关键动作触发的刷新用Full，Fast之后至少接一次Full。保持外部64＋腕部256、原位置、Fast不带state/Full带state；不修改RTC、关键阈值、刷新时间表或FlashRT。只在旧seed7/init0开发，不用seed17输入校准。

扩展原成本探针`--pool-agent`，同三份seed7保存输入、每模式每频率6次，实测高频Full/Fast468.8/329.6ms、10.945/7.704J，中频642.8/463.2ms、10.166/7.177J。Full/Fast各一次作者采样体对照差0，频率与恢复通过；闭环严格使用匹配输入合同的实测成本，不再使用几何proxy。69项相关检查通过；修复三处绕过构造函数的测试替身缺少position_policy初始化，生产适配器没有改动。代码6eed0a3。

一模型完成Spatial0、Object0、Long0各纯Full/候选一次，共六回合，两组均3/3成功、零断供。6次Fast实际发生在Sp0步60、Object0步33/76/152、Long0步81/157；全部符合普通阶段门槛、非连续Fast、state关闭、320token和位置偏移192，所有回合既有审查通过。

|任务|纯Full步数|候选步数|纯Full任务＋在途J|候选J|Fast次数|
|---|---:|---:|---:|---:|---:|
|Spatial0|76|77|67.473|66.663|1|
|Object0|137|209|101.587|136.226|3|
|Long0|223|257|165.153|188.301|2|

全部三个共同成功任务合计334.212→391.191J（增加17.05%），27.032→33.092s（增加22.42%），请求20→23。单次Fast更省，但Object和Long完成过程变长，整任务未省电；只按当前预测判为普通阶段并不足以保证压缩输入不影响后续动作。保留实验入口及全部数据，不替换Full＋Continue，不据三任务推断总体效果。结束恢复1173/3199/1497.6MHz。

[三任务配对报告](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T162003.832561Z-noncritical-fast-dev3-61ea3668/review.md>) · [逐请求准入与汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T162003.832561Z-noncritical-fast-dev3-61ea3668/review.json>) · [匹配成本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T161648.953070Z-dual-selective-fast-36c56aa5/ordinary_fast_costs.json>)。

## 2026-11-06 普通阶段Fast与q20/q10全40扩测完成

同seed17/init1，三组各40任务、共120回合全部完成，结果分别为普通阶段Fast29/40、q20 35/40、q10 36/40。任务＋在途推理5817.540/6664.085/9688.585J，时间473.092/391.584/401.603s，全部既有回合检查通过、零断供。Fast实际74次，均通过普通阶段、非连续、原位置、无state输入记录核对；但相对纯Full动态35/40，成功数下降6项，总能量增加10.74%，共同成功28项仍多耗0.86%，故不替换基线。q20与q10静态组完整保留，不能只按更耗能的参照宣称Fast优势。q20 RAM中断后只补缺少任务，后6项及q10全组使用已有按需词表存储节约1004.5MiB；普通Fast/q20/q10计1/4/1次模型加载。结束频率已恢复，未改FlashRT、RTC、阈值或保护线。

[本次正式报告](FAST_ORDINARY_AND_STATIC_Q_20261106.md) · [120回合与历史参照汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261105T164037.405631Z-fast-q20-q10-all40-cb285719/results.json>)。

## 2026-11-06 两次Full一次Fast：三任务同模型配对

按用户建议增加固定Full→Full→Fast循环，初始Full计入周期；同时保留原Full→Fast循环，默认仍为一对一。新增`--full-per-fast 2`与同模型`--fast-cadence-pair`，只调整刷新模式顺序，不改变KV生成、RTC、关键判断、刷新时间或调频。Full带state，Fast外部64＋腕部256、无state、保留原位置。运行计数只在提交新预测后推进，计划重复判断不消耗次数；独立记录审查按已有请求重建顺序，97项相关检查通过。代码95b7dc9，FlashRT未改。

seed7/init0，Spatial0/Object0/Long0，各两种顺序一次，共6回合一模型，先后顺序交叉。两组均3/3成功，零断供，起点配对与既有回合审查通过，实际顺序匹配配置。

|任务|简单交替步数/J|两Full一次Fast步数/J|
|---|---:|---:|
|Spatial0|77 / 62.537|77 / 66.484|
|Object0|208 / 134.181|133 / 93.959|
|Long0|268 / 193.902|254 / 187.137|

任务＋在途推理总能量390.620→347.579J，下降11.02%；总任务实耗33.564→28.741s，减少14.37%。请求24→22，Fast次数12→7。主要收益来自Object0完成步数减少，Spatial步数相同但能量增加。仅为三任务开发小测，保留显式候选，未替换40任务基线。结束已恢复1173/3199/1497.6MHz。

[同模型三任务配对报告](</home/wyk/workspace/任务产出/perceptbudget/M4/20261106T021500.290570Z-fast-cadence-pair-dev3-2a275826/review.md>) · [逐请求顺序与数据](</home/wyk/workspace/任务产出/perceptbudget/M4/20261106T021500.290570Z-fast-cadence-pair-dev3-2a275826/review.json>)。
