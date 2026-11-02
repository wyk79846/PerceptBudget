# M4 Full高低频闭环配对

首轮Spatial0、Object0各高低频一遍，共4回合、一次模型加载；前者高→低，后者低→高。seed7/init0，H50/N5/BF16/state/RTC及11/12版关键等待、阶段去重不变。不加入Fast。

高频组为Full高频、Continue按原规则降频；低频组在同一规则上允许普通阶段赶得上的Full低频运行。两组动态观测和轨迹可以不同，不声称固定相同输入的频率微基准。

## 计量

- 任务时间仍截止成功/超时判定，不包含3秒尾段。
- 任务结束，先等在途推理完成，然后确认GPU816/EMC2133低频，保持3秒；等待期间不执行环境动作，也不调用模型。
- 返回LOW加固定尾段单列Terminal LOW tail，不算Continue。保存任务能量、任务＋在途推理能量、尾段能量与总能量。
- 计量在LOW尾段结束时截止，之后恢复设备初始频率的管理开销不计入比较；仍保存恢复检查。原入口不带新开关时保持原计量。
- 固定尾段用于覆盖功率读数尾响应，不作为任务耗时或额外节能收益；这是不同长度真实任务的统一结束处理，不是两条等长4秒微基准。

## 执行

在项目目录执行下面命令，输入sudo密码；Ctrl+B后按D脱离tmux，SSH断开不影响测试。

```bash
tmux new-session -s pb-full-pair 'bash scripts/run_m4_ssh.sh --acceptance-group m3 --low-full --full-frequency-pair --task-indices 0 3 --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh'
```

输出仍在任务产出/perceptbudget/M4。完整批次新增full_frequency_pairs.csv/json，保留每回合原始功率、请求步号/RTC前缀、终态时间戳、初始条件配对审核与频率审核。报告未执行到的末次推理数量，不能只按有无commit判定是否执行。

本轮代码检查30项通过、命令dry-run确认4回合/1model。GPU实测待普通用户终端输入sudo密码；没有声称已启动或取得节能结果。修改前备份archive/before-common-low-tail-20261102。

## 扩展剩余10任务

首批602d378f四回合完成，保留Spatial0/Object0原始记录。现补Spatial3/6、Object3/6、Goal0/3/6、LIBERO10 0/3/6，每任务两组，共20回合、一次模型加载。模型、RTC、刷新判断和统一LOW计量保持不变。

已在tmux时执行：`bash scripts/run_full_frequency_remaining.sh`。其参数已通过dry-run检查，需终端sudo密码，准备入口不代表已启动GPU。

新批次full_frequency_pairs.csv仅列新增20回合；combined_frequency_pairs/full_frequency_pairs.csv合并此前4回合，完整24回合；sources.json保存两个批次来源。开始前拒绝重复任务和模型/任务清单不一致的来源，避免重复计算已有结果。
