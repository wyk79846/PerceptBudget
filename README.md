# PerceptBudget

面向Orin NX的VLA时限与能耗调度。任务依据为[Word任务书](../任务指导与相关参考/Student_A_PerceptBudget_Revised.docx)。M1测量已交付，下一阶段为M2设备标定；尚未启动调频实验。

## 唯一基线

配置入口为[config.json](config.json)，约束在[baseline.py](perceptbudget/baseline.py)。

| 项目 | 固定值 |
|---|---|
| 权重 | LeRobot pi05_libero_finetuned_quantiles_v044，revision b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd |
| 预测长度／去噪 | H=50／N=5 |
| 精度 | BF16权重，保留当前FP32动作求解与RTC引导路径 |
| state | 启用；8维归一化后补至32维，文本容量200 |
| 衔接 | 完整VJP RTC、真实异步；beta=5，延迟按实际历史更新 |
| 推理后端 | FlashRT编码器＋OpenPI专家，source-step CUDA图、条件缓存、Flash注意力、融合norm、autotune |
| q | 默认5；扫描5、10、20、30、40；q是两次推理启动的最小控制步间隔 |
| 控制 | 名义20Hz，记录实际频率；推理期间不人为减半运动，不暂停仿真 |

先检查配置和启动参数（不加载模型）：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/check_setup.py
/home/wyk/venvs/lutvla/bin/python scripts/run_baseline.py --dry-run
```

单任务基线测试（会实际使用GPU）：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/run_baseline.py --suite libero_spatial --task-id 0 --q 5
```

正式单任务入口只有`run_baseline.py`。`diagnose_rtc_handoff.py`是其受保护的内部执行器，也供同步/固定延迟正确性诊断使用；诊断不是另一套生产基线。`eval_rtc_openpi.py`与`eval_q.py`只保留公共函数，不再提供旧独立运行命令。当前入口使用已保存的参考run与正确性gate，路径见配置，不能删除这些证据依赖。

## 目录与职责

- `perceptbudget/`：当前推理实现、设备/功率基础、公共评测与分析函数。
- `scripts/`：基线入口、M1复现和分析、正确性审计、环境检查；旧性能探针已清理。
- `tests/`：当前功能、数值参考、日志和供给时序检查。
- `patches/`：现有加载与分词器补丁，保留版本记录。
- 生成结果统一在`/home/wyk/workspace/任务产出/perceptbudget/`，不写进代码仓库。

历史实现代码存于Git分支`archive/pre-single-baseline-20261021`；更早的SEAM/硬前缀实现存于`archive/pre-m2-cleanup-20261021`。历史文档统一在[M1相关报告](../任务指导与相关参考/PerceptBudget归档/M1相关报告/)，不作为当前运行指令。

## M1结果与复现

[最终交付报告](../任务指导与相关参考/PerceptBudget归档/M1相关报告/PerceptBudget_M1阶段交付报告_新模型H50_20261021.md)；[完整主扫描结果](../任务指导与相关参考/PerceptBudget归档/M1相关报告/PerceptBudget_M1主扫描结果_新模型H50_20261021.md)。

保留`run_m1_h50.py`、`complete_m1_h50.py`、`analyze_m1_h50.py`与`analyze_m1_h50_stages.py`供复现。原300次批次已完成，不重新启动或用新源码续写旧批次。主扫描历史逐任务隔离加载，仅作为原实验复现，不作为后续批量测试范式；新测试优先常驻模型。

M1名义动作余量不等于已验证安全缓冲；节能机会估计不等于已实现节能。板载VDD_IN记录称模块输入能量。M2保持模型与算法基线，新增感知档位和设备控制须单独验证。

## 验证

M2首步入口：`scripts/profile_device.py`（只读接口快照）；`--sensor-check`仅执行合成GPU忙闲阶跃，用于传感器检查，不加载模型、不改变频率。结果在外部`M2/`目录。EMC当前需要管理员权限，正式频率组合标定尚未开始。Fast已做固定输入计算短测，尚无真实异步闭环速度或质量结论。

M2固定输入短测入口：`--fast-probe --source-run <成功的H50 Spatial0 q5记录> --repeats 5`。这是隔离的M2视角实验，不覆盖Full基线。20261021短测Full/外部单视角/腕部单视角为461/327/326ms，实际视觉token减半，尚无单视角闭环成功率结论；不把这一数值当完整服务L。

20261021已完成task0三个集合的9条真实异步校准：Full3/3、外部单视角1/3、腕部单视角2/3，均无断供，9条记录独立审计通过。两种单视角均未通过初筛，未进入留出验证；不能将Fast标为可用。单视角6回合常驻模型仅加载一次，Full仍有跨任务内存保护问题，未宣称全面修复。结果在`M2/20261021T094148.675320Z-fast-quality-b689fe6c/summary.md`，本轮服务已结束；退出状态文件保留，桌面通知因本机服务错误发送失败。

```bash
/home/wyk/venvs/lutvla/bin/python -m pytest -q
```

GPU专项检查在无CUDA环境下会跳过，不能把CPU回归通过写成新一轮GPU闭环验证。FlashRT固定为配置所指工作区，本轮清理不修改其源码。
