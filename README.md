## M4联合调度小测（待设备验证）

入口：`python -u scripts/run_m4.py`，默认Spatial task0一次、一次模型加载。
`--dry-run`只查看计划；通过单任务检查后再用`--suites libero_spatial libero_object libero_goal`扩展。
完整候选包含一次刷新和同窗口等待，Continue计入后续Full；Full/Fast按计算频率分别记录RTC延迟。
结果统一放在任务产出/perceptbudget/M4，未修改FlashRT或M3默认配置。

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
