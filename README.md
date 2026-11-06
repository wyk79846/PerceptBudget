# 当前工作锚点（2026-11-06）

按用户决定，保留 **Full→Fast直接交替、33/40成功** 的版本作为后续工作参考。Full→Full→Fast的32/40版本不替换它。

- Full：完整双视角512视觉token，启用state。
- Fast：外部64＋腕部256，共320视觉token，保留原位置编号，不使用state。
- 沿用H=50、N=5、BF16、RTC真实异步、机制一动态刷新和三频调度；本次只记录工作锚点，不改历史实验或通用入口默认值。
- 固定记录：seed=17、init=1、40任务各一次、一次模型加载；327次刷新（177 Full、150 Fast），平均刷新间隔23.52步，零断供。

| 同40任务，包含失败尝试 | 成功 | 任务＋在途推理能量(J) | 全部尝试能量／成功数(J) |
|---|---:|---:|---:|
| 当前Full→Fast | 33/40 | 5064.75 | 153.48 |
| 固定q30高频异步 | 33/40 | 5806.30 | 175.95 |
| 纯Full＋Continue对照 | 35/40 | 5253.25 | 150.09 |

当前版本相对q30能量低12.77%；Continue占当前总能量45.79%，刷新及恢复窗口合计54.21%。相对纯Full＋Continue，总能量较低，但每成功任务能量尚未占优，后续同时保留该对照。

[原始运行计划](/home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153705.566883Z-fast-pos-nostate-seed17-all40-6658feb2/plan.json)保留这版实际使用的成本先验与完整命令，重现时不静默替换为后来标定的成本表。[逐任务结果](/home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153705.566883Z-fast-pos-nostate-seed17-all40-6658feb2/results.md)与[原始数据](/home/wyk/workspace/任务产出/perceptbudget/M4/20261105T153705.566883Z-fast-pos-nostate-seed17-all40-6658feb2/results.json)为此工作锚点的依据。

M4已有动作前瞻、时限与设备准备、Fast闭环实现及实验。按Word要求，仍需围绕当前版本核对同窗口总能量联合选择，补齐同触发器消融、预测提前量与调度开销统计，再形成统一交付报告；固定Full/Fast轮换本身不等于已经验证联合择优。大规模静态前沿、紧功耗窗口和主要收益评测属于M5。

以下为按日期保留的阶段历史；其中“当前”“下一步”均指当时状态，以本节工作锚点为准。

# 历史状态（2026-11-03）

用户当前决定暂不加入Fast，主线继续现有Full＋Continue，Fast实验开关保持关闭。

新增[短空档降频候选](docs/SHORT_CONTINUE_THRESHOLD_20261103.md)：`--idle-min-ms 100`下探200ms门槛，短空档复用原变挡预算，默认仍200ms。相同动作段实测少0.313J；修正版Object3成功157步、0断供，但同批原组超时且门槛生效前已分叉，暂不报告整任务节能率。75项检查通过，全部测试已结束并恢复频率。

后续采用`--idle-min-ms 50`进行[12项验证](docs/IDLE50_12TASK_RESULTS_20261103.md)：11/12成功，零断供，4次短于200ms的降频、其中1次短于100ms。对历史200ms路线共同成功10项省1.12%、时间+0.11%；对异步q30共同成功9项省10.33%、时间+7.98%。11471项记录检查与功率/频率复核通过。首批RAM保护后保留7项、另一次常驻续完5项，实际2次模型加载，全部恢复；未改推理代码或FlashRT。50ms路线保留，原200ms入口仍作对照，40ms暂不采用。

按用户要求新增[Full按总能量选频](docs/FULL_ENERGY_SELECTION_20261103.md)，`--full-energy`启用，原中频优先默认保留。75项检查通过；旧103次决策仅1次改档，Object3有效配对两组成功，131.077→131.036J基本相同。未改相机、Fast、RTC或余量，未重测q30；本轮暂无明显新增节能收益。

当前衔接与Full＋Continue方案保留。新一轮q30/动态各12任务已完成，成功9/12与10/12，共同成功8任务动态能量少11.1%、耗时多5.4%。能耗优化新增[双路Fast小测](docs/FAST_DUAL_SELECTIVE_20261103.md)：两路各160视觉token、不训练；单次热刷新约320ms/7.87J，比Full少35%能量，但三任务闭环两组各2/3、成功任务不同，尚未替换正式默认。

随后实现[普通阶段单次Fast](docs/FAST_ORDINARY_INTEGRATION_20261103.md)：保留三频Full刷新时机与原关键规则，`--ordinary-fast`启用，Fast后必须Full。新批同三个任务原流程3/3、候选2/3，共同成功两例仅省0.61%；6回合1模型、137项CPU检查及独立记录/频率检查通过，零断供。候选暂不作为默认，原Full＋Continue基线保留。

当前`run_m4.py`的Full＋Continue方案（`--acceptance-group m3`）与普通q30都默认开启精简采集：无在途推理时暂停相机，刷新前补采最新双图，state及逐步动作/状态记录保留。`--full-capture`恢复旧采集。FlashRT与RTC实现未改；两侧各一个Spatial0配对已通过，数据见[采集优化记录](docs/M4_THREE_FREQUENCY_TRIAL_20261102.md)。归档的M1/M3入口与旧结果口径不改，后续策略对比使用相同采集设置。

# 阶段记录（2026-11-01）

以11/12版M3为对照。`--low-full`现使用完整剩余窗口安排Continue＋推理：低频少等待，赶不上用高频；关键点与阶段去重保留，结果完成即正常接续，不新增固定交接。155项CPU检查通过，尚未新GPU闭环；原slot_v3锁定窗口导致0准入的结论已撤回。[当前接入](docs/M3_LOW_FULL.md)。以下为阶段历史。

# 当前工作：M3阶段去重已完成12任务，接入低频Full实验

当前版本12任务成功11个，无断供；[结果](docs/M3_PHASE_REFRESH_12TASK_RESULTS_20261031.md)。新增`--low-full`按关键阶段和完整服务时限选择高/低频Full（普通阶段低频赶得上就用，能量仅记录），Fast仍关闭，未自动替换默认。[实现与测试命令](docs/M3_LOW_FULL.md)。CPU检查通过，真实闭环待测。以下回退记录保留为历史。

用户决定暂停Fast接入，先恢复M3。SSH批次入口默认选择`m3`：Full＋Continue、原版机制一、服务时间P95＋25ms余量，推理高频、允许等待时降频。关闭Fast、关键区间额外限制及Full响应规则。RTC、模型、H50/N5与state不变；保留已经验证的内存释放和环境缓存。

回退前代码保存在`archive/before-return-to-m3-20261031`。Fast历史代码与结果保留为实验，不续测本次剩余三个长任务。此前320 Fast说明如下，仅作历史记录。

Fast恢复为外部64＋腕部256视觉token、不带state；Full保持512＋state。416实验不作为默认。旧q5连续交替参考入口：

```bash
python scripts/eval_expert_recompute.py --mixed-pooled-alternation --fast-no-state --cross-suite --dry-run
```

动态调度候选：`scripts/run_m4_ssh.sh --candidate m3_fast --fast-advance`，附原有baseline-run等参数；不要加`--fast-selective416`。新增`--fast-key-phases`实验开关：保护Fast准入，并在当前Fast发现关键区间时提前或立即Full一次；Full提交后不反复触发。Full响应修复尚待Long6单次闭环。依据[PACE论文](../任务指导与相关参考/PACE_2606.00537v3.pdf)核查关键阶段，不自动启动上述测试。

具体下一步见[关键阶段改进](docs/KEY_PHASE_NEXT.md)。以下为项目与历史阶段说明，新工作以本节为准。

# PerceptBudget

面向Orin NX的VLA时限与能耗调度。依据[Word任务书](../任务指导与相关参考/Student_A_PerceptBudget_Revised.docx)推进。**M1、M2已交付；M3最小闭环及三类看门狗设备验证已通过，当前准备四组正式比较，尚未交付M3。**

## 唯一推理基线

设置以[config.json](config.json)和[baseline.py](perceptbudget/baseline.py)为准，正式入口为[scripts/run_baseline.py](scripts/run_baseline.py)。

| 项目 | 设置 |
|---|---|
| 模型 | LeRobot pi05_libero_finetuned_quantiles_v044；revision b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd |
| 预测长度、去噪 | H=50、N=5 |
| 精度 | BF16权重；既定FP32动作求解和RTC引导 |
| state | Full启用，文本容量200 |
| 动作衔接 | 完整VJP RTC、真实异步，beta=5 |
| 后端 | FlashRT感知上下文＋OpenPI专家；source-step图、条件缓存、Flash注意力、融合norm、autotune |
| 控制 | 名义20Hz，记录实际频率；推理期间继续推进仿真 |

诊断执行器及保存的参考run/gate是当前依赖，不作为另一套默认基线，也不能删除证据目录。Fast保持为隔离候选，不覆盖Full默认。未经确认不修改FlashRT、模型、精度或去噪步数。

## 阶段交付

- **M1**：[统一交付报告](../任务指导与相关参考/PerceptBudget归档/M1相关报告/PerceptBudget_M1阶段交付报告_新模型H50_20261021.md)。[历史报告目录](../任务指导与相关参考/PerceptBudget归档/M1相关报告/)。
- **M2**：[统一交付报告（含27项小表）](../任务指导与相关参考/PerceptBudget归档/M2相关报告/PerceptBudget_M2阶段交付报告_20261025.md)。[归档索引](../任务指导与相关参考/PerceptBudget归档/M2相关报告/归档索引.md)。操作接口和恢复见[调频说明](docs/M2_CLOCK_CONTROL.md)。
- **M3进行中（以此条为当前范围）**：[M3最小闭环任务书](../任务指导与相关参考/PerceptBudget归档/M3相关报告/PerceptBudget_M3最小闭环任务书_20261026.md)。当前用Full＋Continue、原版机制一与剩余缓冲，结合完整服务耗时和余量计算启动时刻；机制一进一步优化后置，去噪及混合保留对照。最小闭环和看门狗已验证，接下来统一四组比较。动态Fast和完整联合搜索留待后续。

M2已有27格成本覆盖，重点9格各60次，其余Full/Fast每格20次、高频/均衡Continue每格5窗。刷新能量是相对静态等待的平均增量，Continue能量是执行窗绝对能量，两者不能直接混算；这些不是整任务节能结果。200ms/1s是空闲标定点，不是运行时固定等待指令或已证实的GPU断电状态。

M3开始按相同任务、初始状态和计量范围比较**整任务能量、成功率、完成时间及断供**；需要时补充J/执行步。不为统一口径重复M2测量。动作事件是刷新提示，不是动作可靠性的保证。

## 项目结构

- `perceptbudget/`：推理、RTC、设备控制、功率记录与公共分析。
- `scripts/`：基线、阶段标定、正确性诊断及分析入口。
- `tests/`：功能、数值、记录及供给时序检查。
- `patches/`：既有FlashRT加载与tokenizer补丁及说明。
- `docs/`：必要的操作和恢复说明。

生成结果统一放在`/home/wyk/workspace/任务产出/perceptbudget/`，不上传模型、数据或大体积运行输出。阶段报告统一放在`任务指导与相关参考/PerceptBudget归档/`。新批次优先常驻模型，保留内存保护、GPU锁和异常记录。

## M3当前入口

用户已确认采用原版机制一。`m3.json`的`formal.trigger_strategy=events_only`集中控制正式选择：B/C/D用夹爪、减速、转向提示刷新，关闭去噪触发和事件合并。模型仍执行5步去噪生成动作；模型、RTC、精度、state及看门狗保持原值。

机制一四组入口f447cfd0已通过：四组均成功、0欠载、初态精确配对及频率/功率审核通过。当前可去掉--comparison-screen执行正式36回合。下面保留四组短测复现命令，四回合、一次模型加载：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m3.py --mode comparison --comparison-screen
```

A固定q5高频；B机制一高频；C机制一独立降频；D机制一联合设备准备。新配置短测通过后，去掉`--comparison-screen`跑三集合各task1/3/6，共36回合、一次加载并轮换组序。旧84241059四组短测是单去噪，不作为新配置已经验证的证据。

全部检查初始配对、源索引、Full高频、完整任务功率和频率恢复，生成`comparison.csv`、`comparison_summary.json`、`comparison.md`。任务总能量包含失败尝试与收尾计算；初始不严格一致的任务单独标记。[任务书](../任务指导与相关参考/PerceptBudget归档/M3相关报告/PerceptBudget_M3最小闭环任务书_20261026.md)与[阶段进展](../任务指导与相关参考/PerceptBudget归档/M3相关报告/PerceptBudget_M3阶段进展_20261026.md)保留历次对照及复现入口，历史状态以其日期为准。

## 检查和运行

以下两条仅检查设置和启动计划，不加载模型：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/check_setup.py
/home/wyk/venvs/lutvla/bin/python scripts/run_baseline.py --dry-run
```

单任务基线（实际使用GPU）：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/run_baseline.py --suite libero_spatial --task-id 0 --q 5
```

测试：

```bash
/home/wyk/venvs/lutvla/bin/python -m pytest -q
```

无CUDA环境下的GPU专项跳过不等于真实闭环验证。sudo仅供有界调频助手使用，不用root加载模型。历史实验已完成，复现入口保留但不需要重新执行；原始失败记录也保留。

## Git与恢复

当前开发分支为`project/perceptbudget`，[GitHub仓库](https://github.com/wyk79846/PerceptBudget)。M2报告整理前已保存`archive/pre-m2-report-archive-20261026`。历史实现保存在相应`archive/*`分支；历史说明不覆盖当前配置或最新授权。

DVAC记录小实验已完成：五份已保存Spatial0输入、30组记录开关配对动作差值均为0；专家单独重放约158ms，记录开销未见明显增加。候选前缀受RTC引导和统计范围影响，目前只记录、不参与控制。数据：外部M3/20261026T131006.072736Z-dvac-probe-ce14dd73；详细结论见M3阶段进展报告。

DVAC三任务验证（10月27日）：Spatial/Object/Goal各5份保存观测，一次加载完成。全部5步波动与后续预测修改的排序相关分别0.415/0.782/0.183；这是预测修改代理指标，尚未用于控制或验证成功率提升。原始结果与图在外部M3/20261026T155944.640299Z-dvac-validate-7cece092，进展报告已汇总。

DVAC提前刷新闭环小测已完成：`run_m3.py --mode dvac-paired`在同一次模型加载中配对三任务，共6回合；两组各3/3成功、均无断供。新规则触发3次请求启动，但原20步上限未放宽，请求数和整任务能量基本持平。仅保留为实验开关，不更换默认策略；详见M3进展报告及外部M3/20261026T160829.286738Z-dvac-paired-d5144c8c。

H50覆盖上限对照已完成：`run_m3.py --mode dvac-horizon-paired`在一次模型加载中对比三任务×20/50步上限。两侧均DVAC开启、高频、保留2秒观测年龄检查；H50三任务全成功且无断供，整任务能量分别低13.1%/30.9%/15.5%，Goal耗时增加9.9%。20步对照Object/Goal各15步因上限拒用而缺供，表中保留；H50实际最远用到第30/37/39步，尚未证明连续50步均可靠。默认未自动替换。证据：外部M3/20261027T025834.544642Z-dvac-horizon-paired-80fb474b及M3进展报告。

三种触发策略对照已完成：`run_m3.py --mode trigger-comparison`统一H50上限，在一次模型加载中跑三任务×仅关键动作/仅去噪/取更早融合，共9回合。两单独组各3/3成功，融合2/3，均无断供；仅去噪本轮能量较低。Goal融合超时，但无关键动作主导请求，核对显示与仅去噪从相邻一步观测时机开始分叉，不能单次判定融合必然更差。默认未替换。数据与CSV：外部M3/20261027T031611.450358Z-trigger-comparison-4d8a91d8，完整结果见M3进展报告。

Goal复测（053b6539）已审核：仅去噪/融合各3次均成功、0断供，平均时间8.243/8.209秒、整任务能量150.532/152.989J。融合仅多一个任务结束前未提交的关键动作请求，未见额外收益；前次超时未复现。原始结果在外部M3/20261027T042208.090844Z-trigger-comparison-053b6539，进展报告已更新。


当前M3下一步：仅去噪＋H50接回调频，三集合各task0一次，模型只加载一次。入口和检查已准备，真实小测待用户终端执行（内部sudo提示只用于调频助手）：

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m3.py \
  --mode trigger-comparison --strategies variance_only --device-dvfs
```

`--dry-run`只显示3回合计划，无sudo/模型/频率修改。实际运行先无模型预检，退出核对RESTORED；必要时用既有`sudo /usr/bin/python3 scripts/control_device_clocks.py restore`恢复。机制一只记录，正式基线与FlashRT未改；全量577项检查通过。M3任务书已同步最新范围。

仅去噪H50调频三任务c4b1faa3已审核：3/3成功，能量90.978/122.331/141.577J；Full高频采样匹配、任务退出恢复、原始能量积分均通过。Spatial有1步观测年龄超过2秒而拒用，首轮完整恢复服务实测634ms高于初始553ms预算，需校准后复测；Object/Goal零欠载。不可将当前小测标为M3验收完成。完整审核与CSV在外部M3/20261027T045030.921015Z-trigger-comparison-c4b1faa3，阶段报告已更新。

Spatial复测前已校准首次恢复预算：m3.json引用c4b1faa3/recovery_calibration.json的12条已提交恢复样本，初始P95为634.373ms，热路径与50ms余量不变。单任务复测入口在上述调频命令后加`--trigger-suite libero_spatial`，等待用户终端sudo启动；尚无复测结果。51项相关测试通过，原始标定结果作为运行依赖保留。

Spatial恢复预算校准后复测387688e3已完成：80步成功、0欠载、6.312秒、102.937J，校准来源与首次634ms预算生效、时钟恢复及功率积分均核对通过。相较预先高频参考96.718J，本次能量高6.43%，没有体现净节能；当前只确认单次供给复测通过。后续为故障测试和统一配置正式配对，原始审核位于外部M3/20261027T050657.212708Z-trigger-comparison-387688e3。

H50范围的新实验使用`--no-age-limit`关闭未标定的2秒硬截断；H50步数耗尽和去噪提前刷新保留，旧默认2秒用于复现。高频/调频对照均须采用相同开关。关闭后尚未GPU重测，66项相关检查及worker转发已通过。

无2秒限制的Spatial配对复测入口已准备：`run_m3.py --mode trigger-comparison --strategies variance_only --trigger-suite libero_spatial --compare-device --no-age-limit`。两侧同H50、仅去噪、age=null，先高频后动态调频，共2回合/1model，需用户终端sudo。相关检查通过，尚未实际运行；结果将继续归入外部M3目录。

无age Spatial同模型配对525ad217已审核：高频/DVFS均77步成功、0欠载，能量70.475/63.858J（低9.39%），时间5.669/5.931秒（高4.63%）。实际用到第48/49步、观测年龄超过3秒，硬截断确已关闭。该短回合仅两次预测，去噪阈值尚未激活；不能将收益归因于DVAC触发。原始功率/频率/恢复检查通过，详细结果在外部M3/20261027T052255.735779Z-trigger-comparison-525ad217。

LIBERO-Long已接入M3可选校准：`--trigger-suite libero_10`选择task0（两个物体依次入篮），520步上限与本地OpenPI/FlashRT一致。原三集合默认范围不变，73项相关检查和无主模型环境/图像预检通过；完整模型测试尚未运行。将上面的无年龄高频/DVFS配对命令的suite替换为libero_10即可，两回合共用一次模型；需检查去噪阈值实际激活情况。

Long首轮35462fef：高频有效超时520步；DVFS因旧chunk已走41步只剩9目标步、RTC估计d10而异常退出，频率已恢复。已补调度的离散提交上限`origin+H-d`及恢复确认余量，原记录复放验证更早启动，全量590检查通过。未改变RTC延迟或H50，原命令可重试；尚无修复后Long闭环结果。

### M4固定任务验收（两组）

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4_acceptance.py
```

Spatial/Object/Goal/Long各task0/3/6，seed7、init0；先q30高频Full12回合，再当前联合调度12回合。每组模型常驻，共2次加载。现有默认调度，未加入新PACE规则。结果自动汇总到M4/paired-acceptance会话。后续传`--baseline-run <acceptance-q30目录>`只重测joint并复用固定基线；`--dry-run`检查清单而不加载模型。

M4中断续测：`run_m4_acceptance.py --baseline-run <已完成q30组> --resume-joint <中断的joint组>`。核验并保留全部已完成结果（包括失败），仅预检/执行缺失任务，汇总完整配对表。

SSH断开后继续M4测试：先`tmux new -s pb-m4-v2`，在会话内运行`bash scripts/run_m4_ssh.sh --baseline-run <已完成q30组>`。脚本只在tmux普通用户终端下运行；sudo交互授权一次并每60秒续期，退出后结束续期，不改sudoers。按Ctrl+B再D离开；`tmux attach -t pb-m4-v2`返回。总日志在启动时打印的/tmp/perceptbudget-m4-ssh-时间.log，各回合记录仍在M4输出目录。已改变策略的v2需新跑joint12，不能resume旧策略回合。

M3同12任务对照：在tmux中执行`bash scripts/run_m4_ssh.sh --candidate m3 --baseline-run <已完成q30组>`。复用验收与SSH入口，实际关闭m4_enabled、调用原DeadlineRuntime及D_joint规则，只使用Full/Continue，Full计算高频、Continue可低频；机制一原阈值、H50/N5/state200/RTC不变，余量采用M3阶段报告已选定25ms。12回合/一次加载，结果以m3标签与q30比较，产物集中在M4验收目录以便同任务对照；不改归档的M3报告、m3.json默认50ms或FlashRT。

M3先看前三个任务：上述SSH命令追加`--first-tasks 3`，只预检/执行Spatial task0/3/6，复用q30同三任务并生成配对表；不重测q30，不自动扩为12。
