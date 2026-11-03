# 三档频率尝试

## 原方案保留

- Git分支：baseline/full-two-frequency-20261102。
- 基线提交：bbe414fc42f15d282309d5c5574f4ecbff86253a。
- 原高频Full：GPU1173/EMC3199/CPU1497.6MHz；原低频Full和Continue：GPU816/EMC2133/CPU1497.6MHz。
- 机制：shared_deadline_v4、25ms余量、原关键等待及阶段刷新规则，无Fast。历史12任务两组各10/12，逐任务及能量见M4_FULL_FREQUENCY_12TASK_RESULTS_20261102.md。
- 配置和关键代码SHA256清单：assets/full_two_frequency_baseline_20261102.json。已有模型、代码和原始实验不删除，不覆盖。工作区周报改动不属于此基线，不修改它。

原方案复现入口（先短测task0）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --task-indices 0 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

## 新方案与推进顺序

1. 高频1173/3199、中频816/2133负责Full；低频306/2133只负责Continue。CPU保持1497.6MHz。新方案独立接入，原low-full开关和默认行为保留。
2. 先补测306↔816MHz的切换时间。已有低频306→完整高频约17.98ms，不能直接当作低→中频成本；既有中频Full样本也不能冒充从306MHz恢复后的完整服务实测。
3. 按原有关键动作与缓冲边界给出必须完成时间，比较中/高频的恢复＋服务P95＋25ms余量。每次先判断中频，赶得上就用中频，否则高频；关键动作只提供截止时间，不因关键标签跳过中频可行性判断。高频也赶不上则立即启动，并沿用现有关键等待。进入低频前安排恢复时间，不等到截止才判断。
4. 等待段的降频收益必须覆盖切换成本；首版沿用已有最短等待规则，并记录实际切换及能量，不宣称已重新校准最优等待阈值。正在计算时不降到306MHz。RTC前缀与高/中频延迟历史分开保留，不能混用。
5. 接线后先做原方案与三档方案同任务配对，完整记录等待、恢复服务、任务耗时/能量、实际频率及成功。通过短测再扩展原12任务；不靠单任务宣布达到20%节能。

当前状态：基线保存完成；306↔816MHz测试入口已就绪；三档闭环尚未接入或运行，不影响原方案。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/probe_gpu_clock_switch.py --medium
```

普通用户启动，按提示输入sudo；10次往返，不加载模型，EMC/CPU固定，结束恢复。结果在新的M4/*-gpu-clock-switch-*目录。命令到频率读回不等于模型恢复至动作提交的完整服务时间。

## 低中档切换实测完成

运行M4/20261102T122709.206402Z-gpu-clock-switch-6a72818c。固定EMC2133/CPU1497.6，10次往返：816→306MHz中位2.919ms，最大3.052ms；306→816MHz中位2.840ms，最大3.131ms。20次实际频率及非GPU设置检查通过，RESTORED确认，原始日志重算与summary一致。该数据是静态设置至读回，不包括模型驻留恢复/推理/提交轮询，不能直接当作闭环完整L。至此独立低→中频转换约3ms，低→完整高频仍参考既有18ms整套设置。三档闭环尚未接线，旧基线不变。

## 2026-11-03三档闭环接线完成

独立开关--three-frequency，需同时--low-full --acceptance-group m3；原双档命令、默认profiles和算法不变。新控制器perceptbudget/three_frequency.py：逐次先比较中频Full，再比较高频Full，低频306MHz仅Continue，推理在实际频率确认后开始；计算期间不换档。高/中频RTC历史沿用独立队列，关键动作、缓冲截止及阶段刷新仍由原M3计算。原来的“仅普通阶段允许中频”限制不套在此试验上：关键动作已经形成截止时间，仍先检查中频能否赶上；这是按用户2026-11-03确认的规则。

完整预算含服务分位值、恢复、异步确认控制tick及25ms余量。低→中使用约3ms读回先验，高→中涉及EMC改频使用整套降频先验，不能一律按3ms；低→高使用约18ms先验。服务日志明确包含恢复，不另算一遍Continue；静态转换只是初始先验，首轮必须核对实际恢复及完整服务耗时。等待段沿用原最短0.2秒准入，未宣称该阈值已是新三档最优值。

root增加有限runtime3协议HIGH/MID/IDLE/STOP，复用原快照/锁/40W检查/finally恢复；只有这条新路径跳过未变化CPU/EMC的重复设置。非GPU设置不变时306↔816仅写GPU，原双档runtime仍使用原设置流程。闭环主入口、SSH转发、RTC审计和计量日志已接好；无FlashRT或权重修改。

验证：119项相关检查通过；原Spatial0高频和低频两份已完成实验重新独立审计384/406项均通过。新命令及SSH包装dry-run和语法检查通过。尚未新GPU闭环执行，未声称质量或节能提升。

先测Spatial0一个任务：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency --task-indices 0 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

一模型/一回合，开始前先跑三档实际切换预检，原始数据落M4新目录。已有tmux内需要断开SSH时可将python命令替换为bash scripts/run_m4_ssh.sh并保持其余参数。测试后检查是否真实出现deep_idle、中频/高频的各请求选择、恢复耗时、零断供及成功/完整任务能量，再决定扩大测试。

## 首轮Spatial0与配对验证入口（2026-11-03）

父运行M4/20261103T024232.576458Z-acceptance-m3-4566c1c7；回合M4/20261103T024431.045018Z-record-full-375c6dd7。77步成功，6.638秒，零断供、零关键等待。任务期间68.485J；含在途推理结束与原状态恢复77.104J，7.061秒。模型/专家各加载一次，全部计算频率及RESTORED通过，独立审计442项通过；原始VDD_IN重积分与记录差0J。复核写入回合three_frequency_review.json。

实际请求：初始高频；第28/65步中频Full；第76步高频Full。第2份中频结果在75步提交，带来79步松爪新事件；76步发起高频，任务77步已成功，所以这次结果未执行。先前覆盖的是39步闭爪事件，不是同一个事件。未发现事件重复覆盖失效或切换成本重复计入，不能根据事后任务结束去删这次响应。低→中实际助手9.46/8.96ms，静态3ms先验与完整在线控制路径不同，现有额外控制tick覆盖了该开销；不是只按静态3ms宣称完整恢复。

历史原双档Spatial0为76步、6.048秒、任务期间67.783J。新三档任务能量约+1.04%、时间约+9.74%，未取得该单例的能效提升。初始物理/图像/state及首预测raw相同，但这是跨run对照，在线时间、后续观测、初始RTC延迟估计不同；旧实验另外含3秒共同尾段，禁止直接拿其88.484J与新77.104J当作节能收益。

下一步仅扩到Spatial0/3/6，每任务原双档与新三档各一次，共6回合、1model，组序按任务交替。复用已有--full-frequency-pair，和--three-frequency并用时明确比较full_two/full_three；未开启三档时仍为原full_high/full_low。两组均原Full/RTC/模型、同任务校准和初始噪声，严格比较初始state/图像/首预测raw。原双档子配置关闭三档开关，仍用原runtime及原中频准入规则；新组沿用本轮medium-first，不新增事件规则。

两组统一等在途推理结束后回到GPU816/EMC2133，保持3秒作为共同计量终态，随后才恢复设备原设置。任务期间、在途推理、共同尾段分别统计，避免把收尾成本当Continue收益。尾段不作为生产等待时间。初始配对、动作索引、频率或功率检查不通过则保留原始记录并报invalid，不挑结果。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency --full-frequency-pair \
  --task-indices 0 1 2 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

配对入口及相关131项测试通过；dry-run为6回合/1model/无Fast，旧双档入口仍通过检查。新6回合尚未执行，原Spatial0结果不覆盖。

## Spatial0/3/6配对完成（2026-11-03）

原始父运行：M4/20261103T025809.765693Z-acceptance-m3-358d5a4f；三任务各两组共6回合，1model/1expert，组序交替。三对起始物理、图像、state、噪声、首预测原始及解码动作完全一致。6回合均成功、0断供，频率与RESTORED均通过；逐条原始功率重积分误差为0J，独立动作/RTC审计分别406/438/477/465/605/618项通过。复核记录位于父目录three_frequency_pair_review.json。

|Spatial任务|原双档/新三档步数|原双档/新三档任务时间s|原双档/新三档任务＋在途推理能量J|能量变化|
|---|---|---|---|---:|
|0|76/76|6.087/6.313|68.844/74.017|+7.51%|
|3|87/86|7.364/7.129|88.641/77.869|−12.15%|
|6|109/111|8.331/8.990|104.951/101.556|−3.23%|
|合计|两组均3/3成功|21.782/22.432|262.436/253.442|−3.43%|

这里能量包含任务成功后等待已启动推理完成的实际消耗，排除随后返回816/2133及固定3秒计量尾段。任务完成时间增加2.99%。只积分任务执行到成功时，253.557→242.234J（−4.47%）；若采用相同816/2133终态、纳入恢复和3秒尾段的整路线，则331.567→324.369J（−2.17%）。三种口径来自同一批原始记录，不是三轮独立实验，不挑其中比例较大者作结论。

原双档12次Full，其中中频2次；新三档13次，其中中频7次。新方案确实先中频可行性检查，赶不上才用高频；Spatial6的第42步为中频不可行而选高频，实际所有推理频率读回匹配。每回合均有1份预测没有在任务完成前执行，不能把这些请求或能量丢掉。Spatial0三档又多一次末端松爪高频响应，任务能量稍低，但在途收尾使总能量更高。Spatial3原双档有0.409秒关键等待，新三档没有；因此该例收益不能仅归因306MHz等待。

按服务区间与Continue不重叠划分，原双档Full/Continue约123.42/139.02J，新三档约137.10/116.35J。降低等待功率的部分收益被更多中频服务时间及额外刷新抵消。三档有小幅总体收益，尚未达到相对q30省20%的目标；此批只比较原双档，不直接作为q30节能比例。原基线仍保留，不自动替换。

逐回合路径及精确数据：[CSV](assets/three_frequency_spatial_pairs_20261103.csv)。本轮没有再次改动调度机制；之后先据这三对分析额外刷新与服务时间，再决定是否扩大任务。

## 25ms短等独立尝试（2026-11-03）

用户允许少量短等以减少整数步取整造成的过早刷新。新增--short-wait-ms25（实际命令用空格），只允许三档且开启关键等待时使用。原双档及无该开关的三档保留。修改前备份archive/before-short-wait-25ms-20261103。

每档候选先计算完整恢复/推理预算，加上原25ms安全余量，按当前实测控制周期求向下取整后的缺口。仅当缺口大于0且不超过25ms，允许延后相应毫秒数；缺口更大则仍使用原时间判断。不是每次固定晚25ms，也不缩短RTC的delay或改写实际已执行步数。RTC前缀最晚提交和首次30步上限仍取原硬上限，不能用宽限无限延长H50。原25ms余量字段与新增短等预算分别记录。

执行到原关键动作边界（保持阶段去重）或动作队列空时，若预测尚未完成，可暂停仿真和动作消费，等待结果最多25ms。完成后按真实已执行步数提交，禁止通过减少裁剪数、补零或重播旧动作掩盖缺口。超过25ms则结束该实验回合，outcome=short_wait_limit_exceeded，保留已执行轨迹及等待记录；在途推理仍正常收尾并恢复频率。该结果表示预算未满足，不是脚本报错，也不算任务成功。OS调度可能使观察到的超时返回略晚于25ms，实际等待耗时原样记录，不声称硬实时精度。

候选日志记录required_seconds/period_seconds/ceil_steps/floor_steps/shortfall_seconds/allowance_seconds；回合记录short_waits、short_wait_budget_exceeded。所有short_wait区间独立检查没有env.step开始或动作消费；超限后不得再执行该步。完整任务能量仍包含在途推理，不删除尾算费用。

先对Spatial0做原三档与25ms三档各一次，2回合/1model。沿用现有配对开关；本次标签为full_three_strict/full_three_wait25（两组都三档，只改变短等预算）。初始状态/图像/噪声/首预测严格相同，终态统一816/2133+3秒，原双档对照入口继续有效。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency \
  --short-wait-ms 25 --full-frequency-pair --task-indices 0 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

代码/边界/等待超时/独立审计等152项测试通过，旧6回合重新审计全部通过；同输入离线回放出现规划时间变化，但不能推断新轨迹与收益。dry-run确认2回合1model，原余量仍25ms，无Fast。新真实GPU实验未执行；需要用户在终端输入sudo启动。

### 25ms Spatial0配对结果

运行M4/20261103T033811.439691Z-acceptance-m3-1ce5d972，两回合/1model完成。两组初态、图像、state、噪声、首预测完全一致；均成功、零断供，独立RTC/频率/恢复检查通过，原始功率重积分误差为0J。父目录short_wait_review.json保存复核。

|指标|原三档|25ms试验|
|---|---:|---:|
|完成步数|76|77|
|任务完成时间s|6.3775|6.5608|
|任务＋在途推理J，不含固定测量尾段|74.5880|75.4696|
|相同终态及固定尾段总J|99.2479|100.1656|
|推理次数|4|4|
|中频推理次数|2|2|
|实际短等次数|0|0|

25ms试验任务＋在途推理能量+1.18%，完成时间+2.87%；共同终态能量+0.92%。第一份刷新从第27步推到28步，许可的取整缺口为12.098ms，说明计划放宽确实生效；后续请求为64/75步，原组63/75步。两组都没有实际暂停等结果，因此本轮只验证了取整调度分支及完整闭环，不能宣称验证了25ms暂停的真实影响。没有减少Full次数，也没有得到该单例能效收益；不据此否定所有任务，暂保留独立开关但默认0，未替换原三档或原双档。

逐回合原路径和数字：[CSV](assets/short_wait_spatial0_pair_20261103.csv)。
