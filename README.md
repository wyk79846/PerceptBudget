# PerceptBudget

面向Orin NX的VLA时限与能耗调度。依据[Word任务书](../任务指导与相关参考/Student_A_PerceptBudget_Revised.docx)推进。**M1、M2测量已交付；下一阶段为M3最小闭环，尚未启动新调度实验。**

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
- **M3待实施**：先用Full＋Continue，按最新讨论提前接入未来动作提示，结合缓冲、设备状态相关的完整服务耗时和安全余量计算启动时刻；验证降频/恢复、RTC衔接与异常回退。动态Fast选择及完整联合搜索后续再加入。

M2已有27格成本覆盖，重点9格各60次，其余Full/Fast每格20次、高频/均衡Continue每格5窗。刷新能量是相对静态等待的平均增量，Continue能量是执行窗绝对能量，两者不能直接混算；这些不是整任务节能结果。200ms/1s是空闲标定点，不是运行时固定等待指令或已证实的GPU断电状态。

M3开始按相同任务、初始状态和计量范围比较**整任务能量、成功率、完成时间及断供**；需要时补充J/执行步。不为统一口径重复M2测量。动作事件是刷新提示，不是动作可靠性的保证。

## 项目结构

- `perceptbudget/`：推理、RTC、设备控制、功率记录与公共分析。
- `scripts/`：基线、阶段标定、正确性诊断及分析入口。
- `tests/`：功能、数值、记录及供给时序检查。
- `patches/`：既有FlashRT加载与tokenizer补丁及说明。
- `docs/`：必要的操作和恢复说明。

生成结果统一放在`/home/wyk/workspace/任务产出/perceptbudget/`，不上传模型、数据或大体积运行输出。阶段报告统一放在`任务指导与相关参考/PerceptBudget归档/`。新批次优先常驻模型，保留内存保护、GPU锁和异常记录。

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
