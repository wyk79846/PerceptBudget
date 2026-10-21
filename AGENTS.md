# 项目协作与当前状态

## 唯一有效基线

- 以`config.json`及`perceptbudget/baseline.py`为准：新LeRobot LIBERO H50权重、N5、BF16、state启用/容量200、完整VJP RTC真实异步、beta5、source-step图/条件缓存/Flash注意力/融合norm/autotune。
- 正式入口`scripts/run_baseline.py`，默认q5；扫描q5/10/20/30/40。不要恢复旧模型、运动减半、硬前缀或SEAM为默认。
- 参考run和gate路径见配置，现有诊断内部仍读取这些证据。不要删除原始结果；若要解除历史run依赖，作为独立实现变更审核。

## 已完成与待推进

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
