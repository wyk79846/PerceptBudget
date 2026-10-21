# 项目协作与当前状态

## 唯一有效基线

- 以`config.json`及`perceptbudget/baseline.py`为准：新LeRobot LIBERO H50权重、N5、BF16、state启用/容量200、完整VJP RTC真实异步、beta5、source-step图/条件缓存/Flash注意力/融合norm/autotune。
- 正式入口`scripts/run_baseline.py`，默认q5；扫描q5/10/20/30/40。不要恢复旧模型、运动减半、硬前缀或SEAM为默认。
- 参考run和gate路径见配置，现有诊断内部仍读取这些证据。不要删除原始结果；若要解除历史run依赖，作为独立实现变更审核。

## 已完成与待推进

- 2026-10-21优先状态：前resume服务再次在Full Goal0建图时873MiB触900保护，已停止，不能再称运行；已完成Full Spatial0=77success、Object0=142success均audit/0underflow，Fast尚未开始。继续修复后c12afdb启用仅M2的单线程同GPU共享EGL+释放MjrContext缓冲，保持hard reset/禁止eglDestroy。新gate M2/20261021T101001.012984Z-device-discovery-3a32022b六任务各20步物理/raw/contact/success差0，图像最多3pixel差1；不是soft reset（旧soft gate失败弃用）。339pass9skip110subtests。新服务perceptbudget-m2-fast-quality-shared-egl-20261021已启动，日志/tmp同名.log，续同b689fe6c缺项。profile_device退出保存terminal_status.json并尝试notify-send，桌面通知不等于能自动发聊天消息；不得仅凭启动报告仍运行。此服务终止后读summary及日志更新状态，不自动重启旧服务。原始结果保留，不降低保护线/不逐回合重载/不改变基线或FlashRT。共享EGL是否足以解决长运行尚待实测，不宣称已完全修复。

- 2026-10-21 Fast质量小测进行中：批次M2/20261021T094148.675320Z-fast-quality-b689fe6c，q5，task0三suite×Full/agent/wrist校准；Full提前采task3三suite留出，选定单视角后只测该相机task3，总最多15回合。Full/单视角按几何常驻，agent/wrist共用；不逐回合加载。e5f8dec首Full Spatial0=77success且audit通过；Object建图RAM896MiB触900内部保护，未计任务失败。soft reset对照9292215c状态/像素差异，已拒绝。8ab14f53六任务各20步对照：保留hard reset，仅MjRenderContext析构显式con.free、继续禁用EGL destroy；物理状态/raw/contact/success全同，Goal首帧1pixel差1，其他0，44次释放0errors。da4aef2仅M2 gated启用该资源释放，含resume缺项审计/旧summary备份，337pass9skip110subtests；正式基线与FlashRT不改。续测服务perceptbudget-m2-fast-quality-resume-20261021已启动，日志/tmp同名.log，原fast-quality服务已退出，不重启/重复启动。实时看批次full_quality.json/single_quality.json；已完成回合不重跑。累计Full加载计数包含异常续跑，不称严格总2次加载；900/768保护不降低。未获root调频权限，设备设置仍未改。

- M2续进2026-10-21：用户终端确认EMC rate3199000000/min204000000/max3199000000/locked1，dvfs表含3200000000但超出max不得选；当前权限仍需root，无免密sudo。4ba474a增加隔离m2_profile/m2_view loader与profile_device --fast-probe，不改根配置/FlashRT；545feb3补齐CuBLAS确定性环境。结果M2/20261021T092919.503238Z-device-discovery-e49cfb7b：同已存Spatial0输入+旧目标d5s5，每条件5次，Full/agent/wrist=461.36/326.87/325.97ms，平均请求模块J=12.32/8.51/8.68；视觉512→256、容量712→456、valid665→409，两个形状各自raw/VJP parity0，0fallback。Full和single各加载一次，single复用agent/wrist。只是已准备输入的计算短测，非服务L/闭环成功率/动态节能。首9d4e038c CuBLAS初始化失败无有效样本，保留invalid。12个频率候选及用户EMC来源在run目录，仅提案未调频。下一步Fast候选配对质量验证、受限调频及恢复方案；不宣称已选Fast或三个最终频率组合。

- 2026-10-21已启动M2第一步：新增device_control（只读）、device_profile及profile_device入口；备份分支archive/pre-m2-device-profile-20261021。只读发现当前40W模式、GPU min=max1173MHz，CPU policy0/4 min=max1497.6MHz；EMC debug需要root，沙盒外sudo -n仍要求密码，不擅自改sudo/权限/功耗模式。M2/20261021T091851.698669Z-device-discovery-9018a8cf完成7窗合成matmul忙闲传感器阶跃，20ms轮询、0丢样、值变化间隔中位21.90ms（非硬件更新周期），busy约28.7W/后续idle约10.3W非模型P_on/P_idle。源码num_views=1与序列缩短接口存在，未实测Fast速度/质量，FlashRT和唯一基线未改；未选三个组合、未完成27项小表。继续时先解决EMC读取/控制授权并保持恢复方案，不能宣称M2完成。

- M1主扫描300/300：275成功、25超时。异步q5/10/20/30/40成功28/26/24/27/24，均30任务；零欠载不等于硬实时保证。
- 功率标定26.80896W/9.823655W，模块VDD_IN，非独立GPU/外部完整板级功率；机会估计6.3/13.2/35.7/43.4/47.2%，非已实现节能。
- 两任务阶段候选62.55%unknown，安全B_safe尚不能从H50尾段直接推得。完整结论见归档中的20261021最终报告。
- 用户准备推进M2：三感知档位×三个经筛选的GPU/EMC/CPU频率组合×三起始设备状态。先检查接口、测量边界与切换，再建小表；不提前写M3/M4调度或训练模型。
- 原主扫描及常驻收尾全部完成，禁止重启旧full/graphfix/renderfix服务；不将受控延迟结果当真实异步。

## 工程约束

- 只在主项目`/home/wyk/workspace/PerceptBudget`推进；FlashRT为配置指向的`/home/wyk/workspace/FlashRT-upstream`，不擅自修改。
- 修改FlashRT必须先获授权，并维护其`PERCEPTBUDGET_CHANGES.md`；旧已记录补丁不因清理而撤销。
- 结果仅写配置的外部`output_root`；文档归档在`任务指导与相关参考/PerceptBudget归档/M1相关报告`。历史结果和配置快照不可改写。
- 后续测试优先常驻模型，只有异常才重启；原300任务隔离模式只供历史复现，不用于新批次。保留内存保护、GPU锁、失败记录和退出码。
- 不训练、不更换checkpoint/精度/去噪步数来制造收益。不改任务功耗模式与频率，除非已获对应测试授权。
- 保留正确性参考和必要依赖，不按文件名盲删；旧实验入口删除前做引用检查及Git保存。

## 历史与恢复

- 本轮清理前：Git分支`archive/pre-single-baseline-20261021`。
- SEAM/硬前缀清理前：`archive/pre-m2-cleanup-20261021`。
- 完整历史协作说明与旧README已迁入M1报告归档，旧指令不覆盖当前基线。
