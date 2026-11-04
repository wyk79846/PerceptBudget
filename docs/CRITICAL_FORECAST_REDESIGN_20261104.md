# 机制一重设计：从未执行动作估计关键时刻

用户同意先测Spatial9关键阶段高频，若无改善则重做机制一。单任务同模型两回合已完成，初态、图像、state、噪声及首动作逐值相同。

|方案，均保留全部信号与30步上限|结果|任务时间|任务＋在途推理能量|高频/中频Full|
|---|---|---:|---:|---:|
|中频优先|220步超时|14.43s|171.64J|1/9|
|关键阶段高频|220步超时|14.93s|201.62J|7/4|

关键高频确实触发，仍没有救回任务。2670项独立系统及配对/时钟恢复检查通过，无缓冲耗尽；129项代码检查通过。失败候选`2cf5033`保存于`archive/critical-high-candidate-20261104`，运行源码恢复修改前版本，原30步候选保留。

新提取器`perceptbudget/critical_forecast.py`已写好，流程收敛成四步：

1. 输入当前真正尚未执行的动作，以及最后已执行的少量动作作参照。已过期的预测前缀不作为已执行历史。
2. 提取夹爪命令切换、持续运动幅度下降、累计方向变化。保留现有阈值与连续两步确认；逐渐减速和逐渐转向也能产生提示。
3. 保留三类候选，取最近候选的开始步，计算`t_critical = 当前时间 + 剩余步数 × 实测控制周期`。没有候选则返回空，由原调度使用缓冲截止时间。
4. 后续将这个结果接入原启动时间公式，再按所选中/高频的服务时间安排启动。先保持RTC、频率选择和其余执行规则不变，只验证提取器。

夹爪命令切换不等于抓稳；XYZ动作幅度用于表示运动变化，不冒充真实米/秒速度。

第一版先作为独立提取器完成136项代码检查及四份保存轨迹、736个控制时刻的离线回放检查：候选在未执行段内，确认范围不越界，时间按当前实测周期换算。CPU中位0.464ms、p95为0.802ms，是独立CPU测量。此阶段只验证提取器，没有闭环收益结论。

- [高频候选原始结果及检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T055441.736862Z-critical-high-single-05534ba1/summary.json>)
- [提取器离线回放及逐时刻输出](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T060614.299275Z-critical-forecast-prototype-15fad5ba/summary.json>)、[回放代码](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T060614.299275Z-critical-forecast-prototype-15fad5ba/replay.py>)

## 已接入在线调度与三组单任务对照

已增加`m3_critical_forecast`版本接口：空值使用旧提取器，`remaining_v1`使用新版。每份chunk第一次进入调度时，只提取当时未执行的队列，以最后实际执行的两步作参照；该chunk的候选目标步固定，后续筛掉已执行目标，并按最新实测周期更新预计到达时间，避免目标随滑动窗口一直后退。新增记录`critical_forecast`和`critical_time_monotonic`，独立审查从原始动作、真实执行历史与时间戳重算。接入后138项代码检查通过。

用户提出30步上限是否必要，因此Spatial9同模型增加为三组，各一次，固定seed7/init0。三组均使用原RTC、中频优先、低频Continue、全部信号、原阶段去重及完整输入；没有启用已撤回的关键阶段高频策略。

|方案|结果|任务时间|任务＋在途推理能量|Full次数/其中中频|
|---|---|---:|---:|---:|
|旧机制＋30步上限|220步超时|14.50s|187.31J|11/8|
|新机制＋30步上限|120步成功|8.70s|131.49J|8/3|
|新机制，无固定30|117步成功|8.39s|101.95J|6/4|

三组初始物理、双图、state、噪声、首动作逐值相同，系统/新提取器来源重算/频率恢复检查通过，均无缓冲耗尽。新版两组首次刷新在19/16步，旧组为29步。无固定上限组同时取消首次30步限制，仍保留H50有效队列与RTC提交边界；其实际刷新间隔包含36步，确认取消上限确实生效。

在这一任务中，新版取消固定30仍成功，且本轮能量更低，下一阶段优先保留此候选。默认策略尚未替换，也未新测固定q30静态基线；表中“旧机制＋30上限”仍是动态旧机制，不能将它误称为固定q30或据此宣称整体成功率已超过90%。真实异步下后续启动/交接会有步级差异，本轮仅一次每组。

- [三组闭环汇总、输入配对与独立检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T061546.452855Z-critical-forecast-online-9e7e549a/summary.json>)、[逐组CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T061546.452855Z-critical-forecast-online-9e7e549a/results.csv>)
- [原始三回合记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T061941.405033Z-acceptance-m3-f5eefc4f/summary.json>)、[执行入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T061546.452855Z-critical-forecast-online-9e7e549a/comparison.py>)

## 新版无固定30：原失败任务复测

按用户授权，沿用原40任务中动态方案失败的11项名单，使用`remaining_v1`、无固定30步。Spatial9复用上一轮117步成功记录；其余10项各新测一次，一份常驻模型，未中断或重载。seed7/init0、原中频优先、RTC、阈值及模型参数均保持一致，运行期间源码未改。

|集合|本轮成功任务及步数|仍超时|
|---|---|---|
|Spatial|task2：101；task9：117（复用）|无|
|Object|task4：136；task5：131|无|
|Goal|task0：121；task5：149|task6：300步|
|LIBERO-10|task1：236；task2：211；task9：254|task8：520步|

原失败子集恢复9/11，上轮旧提取器＋30步上限为3/11。新增恢复Spatial9、Object4/5、Goal0、LIBERO-10 task1/2；上版已恢复的Spatial2、Goal5、LIBERO-10 task9均保持成功。

11条有效记录（新测10＋复用1）共15785项独立检查通过，包括新提取器来源、实际执行历史、关键时刻重算、RTC及频率恢复；均无缓冲耗尽。当前策略无首次/周期30步限制，最大实际刷新间隔36步；允许原有关键等待，累计6.76秒，不将有等待描述为连续20Hz。整板任务加在途推理能量合计2157.38J、任务时间合计137.36秒，包含两项超时，不含准备、3秒终态尾段及恢复。

这是同一通用配置的旧失败子集复测，不是新旧同模型成对消融；原动态成功的另外29项尚未用新版复测，因此不换算成新的40任务成功率或整体节能率。新版无固定30候选继续保留，默认配置尚未切换。

- [11项结果与独立检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T062940.154276Z-remaining-forecast-failures-9cdcf1da/summary.json>)、[逐任务CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T062940.154276Z-remaining-forecast-failures-9cdcf1da/results.csv>)、[固定名单及复用来源](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T062940.154276Z-remaining-forecast-failures-9cdcf1da/plan.json>)
- [本批原始10回合汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T063010.708240Z-acceptance-m3-a4448304/summary.json>)、[结束频率读数](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T062940.154276Z-remaining-forecast-failures-9cdcf1da/clock_after.json>)

## 新初态40任务复测：seed17/init1

沿用remaining_v1、无首次/周期30步上限，原RTC、中频优先Full、低频Continue、H50/N5/BF16/state及请求时双图均保留。预先固定seed17/init1，40项各一次，全任务阈值相同，未根据本轮结果改规则。

|集合|成功/任务数|成功率|
|---|---:|---:|
|Spatial|9/10|90%|
|Object|8/10|80%|
|Goal|10/10|100%|
|LIBERO-10|8/10|80%|
|合计|35/40|87.5%|

超时5项：Spatial task4（220步）、Object task1/2（各280步）、LIBERO-10 task0/7（各520步）。全部40项初始物理状态及双图哈希均与原seed7/init0不同，原始init1向量也逐项核对与init0不同。48201项独立系统记录检查及各回合频率/恢复检查通过，无缓冲耗尽；仍有实际关键等待。

第一份常驻模型完成前30项后触发RAM保护；第一次续测在调频助手READY等待超时、没有新增完整回合。普通用户侧任务开始前的READY等待由10秒延至60秒并记录启动耗时后，最后10项完成。共3次模型加载，已完成回合不重跑；机制一、RTC及控制阶段时间规则未改。54项相关CPU检查及空载READY/STOP恢复通过，最终恢复1173/3199/1497.6，FlashRT未改。

准备阶段另有一次像素回放拒收（物理/state精确，18像素差1，超过13像素门槛），0模型/0正式回合，记录保留。后续直接复用已验证、按实际XML哈希匹配的静态环境二进制，初态重新读取，未放宽像素门槛或复用历史结果。

本轮是已参与开发的40任务在新初态下的评测，35/40尚未达到90%以上；未测同seed17/init1的q30，不计算相对成功率或节能优势。

- [40项汇总与检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/summary.json>)、[逐任务CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/results.csv>)、[实时进度最终记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/progress.md>)
- [固定计划/核心源码哈希](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/plan.json>)、[初态核对](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/task_initialization.json>)、[启动等待修正](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/clock_ready_fix.json>)
- [完成批次及前30项保留来源](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T072948.508597Z-acceptance-m3-50e62d73/summary.json>)、[运行入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/candidate.py>)、[汇总脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T065926.074258Z-remaining-seed17-init1-40-06d173c6/review.py>)

## 阶段去重优化：共享事件来源，并只覆盖同类事件

检查发现机制一已经使用remaining_v1，但阶段边界仍用旧detect_events重新寻找事件锚点。40任务保存记录中的414个请求离线核对，17次新增阶段覆盖、1次失去覆盖。候选先只统一事件来源，阈值和运动阶段边界保持原状。

仅统一来源的V1不采用：同模型task7旧规则300步成功、V1为520步超时；task6两组均224步成功。V1把第38步转向响应当作38–60步整段已处理，省掉了针对47步减速的35步刷新；两组首次动作差45步、场景差46步。2071bdf源码保存在archive/shared-phase-source-v1-20261104。

修正版V2让阶段与实际等待共用机制一的事件，同时只去重已经响应过的同类信号：转向不覆盖新的减速/夹爪变化，夹爪开合方向相反仍单独处理。没有改阈值、训练模型、改RTC或改FlashRT。新增选项m3_phase_event_source=remaining_v2，需要remaining_v1及phase_refresh；默认保持原规则，旧35/40基线保存在baseline/remaining-v1-seed17-init1-20261104。

修正版另一批同模型4回合，seed17/init1，两任务顺序交叉，初始物理、图像、state、噪声和首动作逐值一致：

|任务|本批旧规则|本批V2|旧/V2任务时间s|旧/V2任务＋在途推理J|
|---|---|---|---:|---:|
|LIBERO-10 task7|520步timeout|272步success|31.183/19.383|547.609/395.399|
|LIBERO-10 task6|520步timeout|236步success|29.190/13.821|484.629/246.746|

V2分别保留了task7第66步转向刷新、task6第44步转向刷新；旧阶段覆盖分别把这些不同于已响应类型的新提示过滤掉。首次动作差在80/54步，首次场景差81/55步，均晚于对应刷新。154项CPU检查、11735项独立系统记录检查、两组初始输入配对及全部频率恢复通过；0缓冲耗尽，本批1次模型加载。V1四条历史记录在新审查器下仍通过。

V1、V2的旧规则跨批结果有波动，分别报告，不跨批挑选基线。这里是两项已知任务的开发小测；V2保留为显式候选，未替换原40任务基线，不把2/2外推成37/40或总体成功率。表中能量随任务提前成功而变化，不等同固定工作量的省电率。完整模型、频率、RTC、源代码及输入证据保留在各实验目录。

- [V1与离线检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T074819.883850Z-phase-source-check-26c3e8e0/summary.json>)、[V1首次差异](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T074819.883850Z-phase-source-check-26c3e8e0/first_difference.json>)
- [V2汇总、配对与干预位置](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T080311.648034Z-phase-kind-pair-63081837/summary.json>)、[四回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T080311.648034Z-phase-kind-pair-63081837/results.csv>)、[固定计划与源码](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T080311.648034Z-phase-kind-pair-63081837/plan.json>)、[运行入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T080311.648034Z-phase-kind-pair-63081837/pair.py>)

## 冻结V2小规模基线与q30：同40任务、seed17/init1

未进一步微调阈值，V2完整40任务为36/40（90%），四套各9/10。超时为Spatial4、Object1、Goal3、LIBERO-10 task8。与前一版本35/40的历史结果相比，Object2、Long0/7恢复，Goal3、Long8退步，净增加1项。全部阈值、模型、N、精度、state、RTC及采集规则冻结。V2前31项后RAM保护，只续测缺少9项，共2次模型加载；原有效结果未重跑。

用户随后指定将V2冻结为小规模对照基线，再测q30。q30为高频Full、每30控制步启动预测、RTC真实异步；同样H50/N5/BF16/state、seed17/init1、请求时双图及图缓存，不带事件判断/DVFS/Fast。用户通过离线tmux执行，40任务常驻1次模型正常完成。

|集合|q30|V2|
|---|---:|---:|
|Spatial|7/10|9/10|
|Object|10/10|9/10|
|Goal|8/10|9/10|
|LIBERO-10|8/10|9/10|
|合计|33/40（82.5%）|36/40（90%）|

q30超时：Spatial1/2/8、Goal3/9、Long0/1。两组各有成功与失败，V2净多3项、成功率高7.5个百分点；不把它说成覆盖q30全部成功任务。

|同口径累计指标|q30|V2|V2变化|
|---|---:|---:|---:|
|全部40次尝试能量J（含超时）|5806.30|7731.48|+33.16%|
|全部40次任务时间s|399.58|442.66|+10.78%|
|全部40次推理请求数|257|541|—|
|同输入、共同成功29项能量J|3347.23|4992.17|+49.14%|
|同输入、共同成功29项任务时间s|231.52|282.62|+22.07%|
|上述29项推理请求数|151|353|—|

本轮V2成功率更高，但没有形成节能收益，不能宣称任务书节能目标已达成。可见的主要代价是刷新次数增加；共同成功子集也出现更高能耗、更长时间，因此不能仅用成功率改善评价系统效率。

能量均为VDD_IN模块输入的正式任务＋在途推理收尾，不含模型加载、环境准备、恢复及统一3秒终态尾段。两组无缓冲耗尽，实际关键等待照常计入任务时间。51942项V2、34005项q30独立系统检查及频率恢复通过；q30逐请求核对确为0/30/60…间隔。

同种子初态仍核对实际输入：40对物理状态/raw state/噪声相同，39对图像哈希相同；Goal2图像不同但两组都成功，从共同成功30项中剔除，得到29项匹配子集。首次raw动作仅9/40逐值相同，最大绝对差0.0139506；属于分进程对照，不能把所有轨迹差异唯一归因调度机制。主统计保留全部80有效回合，无逐任务调参或挑选复测。

- [冻结V2完整40汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T090527.343866Z-phase-v2-all40-406cb8fa/summary.json>)、[固定配置与源码](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T090527.343866Z-phase-v2-all40-406cb8fa/plan.json>)、[V2逐任务CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T090527.343866Z-phase-v2-all40-406cb8fa/results.csv>)
- [q30完整40汇总与审查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/summary.json>)、[q30运行入口](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/candidate.py>)、[q30固定计划](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/plan.json>)
- [逐任务成对表格](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/comparison_with_v2/comparison.md>)、[80回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/comparison_with_v2/comparison.csv>)、[分口径汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T093629.172429Z-q30-v2-reference-all40-ca14601f/comparison_with_v2/metrics.json>)

## 开发样例与测试样例分开

用户澄清：本次仅排除seed17/init1的40个测试样例；任务本身可以复用，开发使用已测过的seed7/init0，不扩到LIBERO-90。下面的修改只看旧组，不使用本轮测试样例调参。此前已完成的开发和测试记录保持原样，不把已看过的结果称为从未接触过的数据。

第一项候选只将减速/转向在未来动作中的持续确认从2步增至4步，夹爪维持2步；事件起点不延后，尾部不足4步时仍保留原2步确认。四套各task0、两种规则各一次，共8回合、1次模型加载，交叉执行顺序，初始物理/双图/state/噪声/首动作四对精确一致。

|旧组开发小测|原V2|持续确认4步|
|---|---:|---:|
|成功|4/4|4/4|
|Full次数|54|56|
|任务＋在途推理能量|754.67J|774.13J|
|任务时间合计|42.01s|41.54s|
|缓冲欠载|0|0|

能量增加2.58%，没有达到减少刷新与能耗的目的，不采用该候选。144项CPU检查和9982项运行记录检查通过；频率已恢复1173/3199/1497.6MHz。候选代码a98c1a6保存在archive/motion-confirmation-candidate-seed7-20261104，三个源码/测试文件恢复910c84b，不在主线叠加无效选项。

另外检查了旧组原V2的等待后事件记录，17次墙钟迟到但控制停在事件边界的请求中，没有发现紧接下一次请求重复响应同类、5步范围内事件，暂不修改该逻辑。

- [旧组开发计划与样例范围](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T123319.362707Z-motion-support-seed7-9e5a8086/plan.json>)、[八回合结果](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T123319.362707Z-motion-support-seed7-9e5a8086/summary.json>)、[逐回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T123319.362707Z-motion-support-seed7-9e5a8086/results.csv>)、[独立审查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T123319.362707Z-motion-support-seed7-9e5a8086/postrun_audit.json>)

第二项候选不增加代码，只将原转向阈值60°调整为90°。夹爪、减速、阶段覆盖、RTC、频率、缓冲边界和采图方式不变。仍为旧seed7/init0四套各task0，两个条件各一次、1次模型加载。每项候选都与本批原规则比较，不用上一批更低或更高的能量作分母。

|旧组task0|原60°→90°步数，均成功|Full次数|整任务能量，含在途推理|能量变化|
|---|---:|---:|---:|---:|
|Spatial|78→77|10→9|140.10→124.10J|−11.42%|
|Object|141→143|11→10|158.24→141.52J|−10.57%|
|Goal|121→199|7→11|102.65→165.68J|+61.40%|
|LIBERO-10|252→288|26→25|368.38→355.33J|−3.54%|
|合计|两组均4/4|54→55|769.37→786.63J|+2.24%|

总任务时间41.46→46.40秒，增加11.92%；均零欠载。四对初始物理、双图、state、噪声、首动作精确一致；10286项系统检查、频率恢复通过，原始功率重积分误差0J。Goal完成时间明显拉长，抵消了其他任务的节能，故不采用统一90°参数；局部省电不作为全局收益。两轮共16回合、2次模型加载，最近seed17/init1的样例未参与新参数开发。当前源码、配置及测试与910c84b一致，冻结V2保留，FlashRT未修改，无后台GPU任务。

本轮结论：简单增加确认时长或统一增大转向角度，均未得到更优的整体结果。后续若继续，应区分普通移动和关键操作的运动变化，再用同一开发组验证；不将本轮失败候选加入默认路径。

- [转向参数开发计划](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T124343.123498Z-turn90-seed7-f927c62e/plan.json>)、[八回合汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T124343.123498Z-turn90-seed7-f927c62e/summary.json>)、[逐回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T124343.123498Z-turn90-seed7-f927c62e/results.csv>)、[独立审查与功率重积分](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T124343.123498Z-turn90-seed7-f927c62e/postrun_review.json>)

## 后续候选：保留尚未到期的各类响应记录

继续检查旧seed7/init0记录，发现V2只保存最近一次阶段响应。一个有限区间内先响应转向、再响应减速后，前一份转向覆盖会被覆盖；再次出现该区间的转向时，原规则可能重新刷新。旧四项54次请求中5次符合此情形，这是离线机会数，不等于闭环必能省掉5次计算。

候选只补齐记录：每份已完成响应保留自己的类型、原始开始/结束步与来源，原区间结束就删除。不同信号仍分别判断，不将转向覆盖扩大成减速/夹爪覆盖，不合并重叠区间来延长旧记录。新选项`m3_phase_coverage_memory=all_active_v1`只支持已存在的V2同类覆盖；默认空值仍复现原V2。关键提取器、阈值、PhaseWait、缓冲与RTC、频率规则均未调整。独立审查从原始提交记录重建全部有效覆盖，缺记录、改边界或超期使用均会失败；续跑也检查版本一致。源代码`4fa47f4`，178项CPU检查通过。

固定旧seed7/init0、四套各task0，原/新各一次，共8回合、1次模型加载，交叉执行顺序：

|任务，均成功|原→新步数|原→新Full次数|原→新任务能量，含在途推理|原→新任务时间|
|---|---:|---:|---:|---:|
|Spatial0|81→80|10→10|137.92→134.00J|6.78→6.63s|
|Object0|142→141|11→11|156.78→157.34J|8.83→8.85s|
|Goal0|121→121|7→7|104.30→103.88J|7.03→7.03s|
|LIBERO-10 task0|272→262|24→19|344.02→279.32J|18.83→16.64s|
|合计|两组均4/4|52→47|743.01→674.54J|41.47→39.16s|

本批总能量下降9.22%、总时间下降5.57%，主要来自长任务（能量下降18.81%），均无缓冲欠载。四对初始物理、双图、state、噪声与首动作精确一致；9665项系统检查、频率/恢复及原始功率重积分通过，能量误差0J。模型保持常驻，设备最终恢复1173/3199/1497.6MHz，FlashRT未改。

候选确实使用了早于最近一份响应的记录：Object0自123步起1条提示、长任务自115步起8条提示；Spatial/Goal没有此类额外覆盖。不过真实异步两条轨迹的首次动作差异分别在Spatial37、Object69、长任务72步，早于新增覆盖起效；Goal动作整段相同。因此这是同批真实部署结果，不能将全部节能严格归因于这一处修改，更不能由四个开发样例推成40任务结论。保留为显式候选，原V2基线保留，最近seed17/init1未参与调参；后续扩大验证仍使用旧开发组。

- [冻结开发计划与源码](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/plan.json>)、[八回合汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/summary.json>)、[逐回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/results.csv>)
- [独立审查与能量重算](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/postrun_review.json>)、[旧响应实际生效位置](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/intervention_review.json>)、[轨迹及观测步数差异](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T130110.702277Z-phase-history-seed7-b9127b84/pair_trajectory_review.json>)

## 扩样：原始历史记忆候选没有通过

用户同意继续后，固定源码补测旧seed7/init0四套task3、6，共16新回合、1次模型加载；之前四套task0的8回合保留，不重复执行。合计12任务24回合、两批均在各自同一模型内配对。新增原版7/8成功、候选6/8成功；唯一新增退化是Spatial3（原86步成功，候选220步超时），Goal6两组均300步超时。

|合并12任务，失败均计入|原V2|保留历史候选|
|---|---:|---:|
|成功|11/12|10/12|
|总能量|2381.83J|2434.97J|
|每成功任务能量|216.53J|243.50J|
|总任务时间|133.96s|141.87s|
|Full次数|168|169|
|欠载|0|0|

每成功任务能量增加12.45%，故不采用该候选。共同成功10项能量虽下降5.34%，不足以支持整体改进；此前四项9.22%的下降未在扩样后保持。两批共32521项系统检查、频率/恢复与原始功率重积分通过，最近seed17/init1没有参与开发。

Spatial3的记录给出一个具体匹配问题：42步转向的响应主要向下，53/54步新的转向已主要向上，与原响应方向相差约145°/134°，却被“同为turn、仍在原区间”压掉；后续89步相似情形约161°。这支持先尝试转向方向校验，不支持直接放宽所有事件阈值。下一候选仍保留原区间截止，只有新旧响应方向相符才当作同一转向；显著差异沿用既有60°标准，不按任务逐项调参。该线索尚不是唯一失败原因的证明。

- [新增16回合汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T133604.968247Z-phase-history-expand-64e90d3c/summary.json>)、[完整12任务24回合汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T133604.968247Z-phase-history-expand-64e90d3c/combined_development.json>)、[完整逐回合CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T133604.968247Z-phase-history-expand-64e90d3c/combined_development.csv>)
- [新增运行独立审查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T133604.968247Z-phase-history-expand-64e90d3c/postrun_review.json>)、[Spatial3方向核对](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T133604.968247Z-phase-history-expand-64e90d3c/spatial3_motion_identity_review.json>)
