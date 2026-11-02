# Continue更低频小测

目的：比较当前低频与更低GPU频率，EMC保持当前2133MHz，确认降低保持耗电后，恢复高频并完成一次Full是否仍有净收益。暂不改正式调度器。

| 档位 | GPU MHz | EMC MHz | CPU MHz |
|---|---:|---:|---:|
| 当前Continue | 816 | 2133 | 1497.6 |
| 候选一 | 408 | 2133 | 1497.6 |
| 候选二 | 306 | 2133 | 1497.6 |

一次模型加载；三档正向、反向各一轮，每档每轮5次，共30个测量请求，不是30个任务。每次先用相同输入推理建立负载历史，再降频等待1秒、恢复高频、完成相同Full RTC推理，统一在降频命令后3.5秒截止积分。测量窗口内至少保留1秒推理后尾段；等待功率取降频读回后200ms至恢复命令之前。该等待功率仍可能受传感器响应影响，不能单独替代整段能量结论。

输出等待功率、恢复命令至动作返回耗时、共同窗口总能量、10对逐次节能差及输出一致性。功率为板级VDD_IN总功率；这是静态驻留模型筛选，没有实时执行动作，不冒充真实Continue或整任务收益。Continue若占50%时间，也不等于占50%能量，整任务节能须后续实测。

沿用原有有限时钟助手、频率读回、快照恢复、GPU互斥及内存保护。仅诊断进程扩展候选，不修改正式时钟表、FlashRT、权重、RTC或任务逻辑。不涉及railgate。频率不被设备接受时，在模型加载之前停止并记录预检日志。

执行（普通用户，不在命令前加sudo）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/probe_lower_idle.py
```

终端输出run_dir；过程查看该目录worker.log。结果为lower_idle.json、summary.json、每轮时钟及原始功率记录。目录位于任务产出/perceptbudget/M4。

需要手动恢复且已有待恢复快照时：

```bash
sudo /usr/bin/python3 scripts/control_device_clocks.py restore
```

CPU检查：test_lower_idle_probe、test_device_clocks、test_device_clock_recovery共35项通过；dry-run及语法检查通过。尚未开展设备频率/功耗实测；当前sudo需要用户密码。

## 首次设备预检与修正

20261102T114048.607933Z-lower-idle-screen-e44fcf5c：当前816/2133档通过；408/1600档申请EMC1600MHz，500ms后实际仍为2133MHz，预检失败。根助手已输出RESTORED；未加载模型、没有推理/能耗结果。不能把申请频率当作实际频率，也尚不能断言1600MHz在所有条件下不可用。本轮将三个档位的EMC统一为已验证可用的2133MHz，仅比较GPU816/408/306MHz，暂不继续尝试665.6MHz EMC。正式配置不变。
