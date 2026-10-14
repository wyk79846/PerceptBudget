# PerceptBudget

在 Jetson Orin NX 上评估 π0.5 的动作刷新频率与任务质量、耗电之间的关系。已有最小评测程序和功率记录工具；CUDA 扩展编译、导入及模型短测已通过。当前 M1.1 仅计划 q=5 同步重测30个不同任务，各一次。此前150次 q 扫描存在执行模式混杂，作为历史记录保留，不能作为正常基准或与本轮合并。尚无真实20Hz闭环或异步节能结论。

2026-10-15 更新：已授权加载内存补丁及分词器缓存补丁，均保存于 `patches/`，哈希记录在配置中；不改模型计算，应称“原版＋已记录补丁”。独立短测验证 BF16、启用 state 和 H=20 输出能力；state 是额外输入实验，8维按权重 q01/q99 归一化、不补维。旧45次初筛以及此前150次 q 扫描均仅作历史记录；150次执行模式混杂，不作为正常基准。实际仿真慢于20Hz，不能据此声称真实20Hz闭环表现。

## M1 协议

此前150次运行记录：[历史 summary](../任务产出/perceptbudget/M1/20261014T145920.598259Z-q-screen-dc24e13a/summary.md)。该轮执行模式混杂，仅供追溯，不作为本轮基准。

当前先完成 **M1.1：q=5 同步重测基准**；确认基准后继续 **M1.2 功率与耗电核验**。后续 q=8/10/12/15 同步质量扫描仍保留在计划中，本轮先做 q=5。

固定目标配置：当前 `pi05_libero_pytorch` 权重、BF16、去噪步数 `N=10`、启用机械臂 state、预测长度 `H=20`，`q=5`。选取30个不同的 LIBERO 任务，Spatial、Object、Goal 各10个；全局 task ID 为0–9、10–19、20–29，开跑前核对 ID 与任务名称。每个任务只同步运行一次，共 **30 次**。固定 seed=7、初态索引0、控制周期0.05秒；步数上限按 suite 分别为 Spatial 220、Object 280、Goal 300。记录失败或超时及实际步数，不将其自动归因于 q。

本轮只建立 q=5 同步基准，不做 q 间比较，也不据此推断异步执行效果或节能收益。按 Spatial、Object、Goal 分别报告成功数/任务数、失败与超时原因，并附总体结果；单个任务分别对应总体约3.33个百分点、suite内10个百分点。

独立短测已验证 BF16、处理后的 state 输入及 `H=20` 输出；正式运行继续记录实际配置和异常。`H=20` 输出不代表任务质量通过。不修改 FlashRT，也不复制旧动作凑长度。`H` 是单次预测动作数，`q` 是执行多少个动作后重新观测并推理。

### M1.1：q=5 同步重测

记录每次运行的 suite、task ID 与名称、q、成功/失败/超时、已执行步数、推理次数和异常现象。输出按 suite 分层的结果表、总体汇总及主要失败原因。实际仿真慢于20Hz，不能把仿真结果描述成真实20Hz闭环表现。

### M1.2：功率与耗电

确认同步基准后继续 M1.2 功率与耗电核验，记录功率覆盖模块还是整板，并使用明确标注的真实运行时间窗口积分。同步重测中的等待计入 wall-clock，所得结果只能描述本次同步诊断，不能冒充异步闭环或据此声称异步节能。没有可靠读数就标记未测。

### 计时与运行模式

- 仿真任务时间 = **正式控制步数 × 0.05秒**。
- 真实 wall-clock 从正式任务开始计至结束，包含推理、等待和调度开销；同步推理暂停仿真的等待不增加控制步时间，但计入 wall-clock。两种时间分别报告，不用单次推理时间替代。
- 本轮标明 **同步诊断**。同步结果不能当作异步闭环结果，也不能据此声称异步节能。
- 成功、失败和超时分别记录；失败不伪造成功完成时间。只在共同成功的相同任务间比较完成时间，并同时报告成功次数。

## 当前代码与运行结果

- `perceptbudget/experiment.py`、`scripts/new_run.py`：读取配置并创建仓库外的独立 run 目录，保存配置快照等管理信息。
- `scripts/check_setup.py`：只读检查路径和配置；字段检查不代表后端已实际使用相应精度或输入。
- `scripts/run_flashrt.py`：在独立进程入口选择配置的 FlashRT 源码。
- `perceptbudget/control.py`：一个推理请求在途的动作队列，按控制步对齐，记录过期动作和断供。
- `perceptbudget/libero_input.py`、`model.py`：双图/state 预处理与原版前端适配，不修改 FlashRT。
- `perceptbudget/power.py`：只读 VDD_IN 采样与积分，标明数据覆盖度，缺读不冒称总耗电。
- `scripts/eval_q.py`：环境检查、单条 smoke 和正式 q 评测入口；本轮协议为 q=5、30个任务、同步运行。
- `config.json` 保存目标配置；正式结果以独立 run 快照和日志为准。旧轮记录独立保留，本轮结果待完成；尚未验证真实20Hz闭环或异步节能。

结果放在 `/home/wyk/workspace/任务产出/perceptbudget/<阶段>/<唯一run-id>/`，不覆盖旧 run。评测时保存配置、代码版本、任务、q、原始轨迹、状态、时间和功率数据；构建产物放在同一结果根目录的 `build/`，环境检查结论单独保留。

```bash
/home/wyk/venvs/lutvla/bin/python scripts/check_setup.py
/home/wyk/venvs/lutvla/bin/python scripts/new_run.py M1 screen
/home/wyk/venvs/lutvla/bin/python scripts/run_flashrt.py your_script.py [args...]
```

单条运行检查（原版扩展构建完成且输入方案确认后）：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/guarded_run.py -- \
  /home/wyk/venvs/lutvla/bin/python -u scripts/eval_q.py --smoke
```

本轮 H=20 与启用 state 是实验配置，不代表官方默认 H=10 且不启用 state 的配置。当前 state 配置为 true；本地 OpenPI `pi05_libero` 的 `discrete_state_input=False`，FlashRT 传入 state 会额外写入 prompt。8 维 state 按权重 q01/q99 归一化、不补维；用户已批准本轮保留该输入。

现场读到 40W mode 4，CPU policy0 的最小/最大频率均为 1497600 kHz，GPU 均为 1173000000 Hz。第一阶段保持现有锁频状态，不自行调整；不能将现状称为默认自动调频。

M1 的顺序是：确认配置与后端能力 → 完成 q 评测 → 核验功率和耗电 → 根据结果实现最小策略。后续只做完成当前实验所需的实现；不要求论文级统计、完整设备标定或大规模重复测试。

## 边界

- 使用固定原版 FlashRT 加上述两个已授权补丁；其余源码不改，实验控制和后续策略放在本项目。
- 不把计划、配置或管理工具描述成已完成评测；节能结论必须有实际功率/能耗记录。
- 若 state 预处理规范未确认，不猜测归一化或自行补维；`state_enabled=true` 本身不能证明 state 已进入模型。
- 保持旧 LUTVLA、共享依赖和现有虚拟环境不变。未经明确安排，不并发运行 GPU 作业。
