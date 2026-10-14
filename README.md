# PerceptBudget

研究感知计算预算与可靠动作执行之间的关系。当前已具备 FlashRT 环境预检查、隔离导入入口和外置实验目录创建工具；尚未运行真实实验，也没有性能或能耗结论。

## 目录与进度

- `perceptbudget/experiment.py`：读取并检查配置、验证结果根目录、创建独立 run 并保存 `run.json`。
- `scripts/check_setup.py`：只读检查本地路径、配置和 FlashRT checkout；验证 `N` 为正整数、精度为 `fp16`/`bfloat16`，默认不导入 torch。
- `scripts/run_flashrt.py`：在独立进程入口中选择配置的 FlashRT 源码，避免旧同名包串入。
- `config.json`：保留设置检查器所需字段。模型、`H=10`、`N=1`、`fp16` 是首个基线前的候选；基线确认后由各 run 的配置快照记录并在同组比较中冻结。state 默认关闭。

阶段按顺序推进，目前均未验收：

- M1：评估 `q` 对质量和轨迹偏差的影响、有效缓冲，以及满载空闲功率机会；`q` 初筛与复核是 M1 子任务。
- M2：建立档位、频率和设备状态表，并完成转换标定。
- M3：完成串行刷新、异步动作衔接和最晚启动的最小闭环。
- M4：实现动作前瞻与联合节能调度。
- M5：以充分调好的公平基线验证真延迟闭环/真机表现及消融收益。
- M6：完成复现与论文交付。

阶段未验收前不以计划代替结果。

## 运行结果

所有生成结果默认写到仓库外的 `/home/wyk/workspace/任务产出/perceptbudget`，按阶段和 run 隔离：

```text
<output_root>/<M1…M6>/<UTC时间>-<用途标签>-<唯一后缀>/
└── run.json
```

`run.json` 保存配置快照、创建时间、用途、代码提交和工作树状态。实验实际产生后，再在对应 run 目录记录 `trace.jsonl` 原始轨迹、`summary.json`/`summary.md` 结论及 `plots/`、`videos/`；创建 run 时不会预造这些空文件或目录。实验记录应区分成功、失败、超时和运行无效，并分别记录仿真动作步数、完整刷新次数、推理墙钟、纯任务墙钟（不含加载/预热/建环境）及仿真时间。没有功率积分数据就不报告能耗收益。

显式创建实验目录（不会加载模型或控制设备）：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/new_run.py M1 screen
/home/wyk/venvs/lutvla/bin/python scripts/new_run.py M2 confirm
```

阶段只能是 `M1`–`M6`；用途可用 `screen`、`confirm`、`final` 或简短字母数字标签。省略用途时默认为 `screen`。已有 run 不会覆盖。

## 重要边界

- 不修改原版 FlashRT、旧 LUTVLA 或现有 venv；不训练、不切分 KV、不改推理逻辑。
- M1 关注可靠执行长度。完整刷新输出 `10×7` 时只消费前 `q` 个动作、丢弃尾段；消费期间不推理，之后再完整刷新。旧 chunked 流程的逐步 `advance` 不是本项目协议。
- Continue 消费已有动作。Fast 尚待验证，不能用旧 KV decode 冒充验证结果。
- 读取机器人状态日志不等于模型启用了 state 输入。同步仿真暂停期间的墙钟也不代表真实机器人完成时间。
- `N=1` 不保证成功率；单次策略失败先排查运行有效性，不据此改变同组基线。旧 LUTVLA 实验不作为本项目能耗证据。
- 原版 FlashRT checkout 位于 `/home/wyk/workspace/FlashRT-upstream`，本地快照 revision 为 `e4fe1232ee49f8d3cd96e32706114fd960b320fb`，不代表上游最新版本。

## 设置检查与导入

使用配置指定的 Python 做只读检查；`--check-import` 会在 CPU 子进程中导入包，但不加载模型或初始化 CUDA：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/check_setup.py
```

同名包通过进程入口选择独立源码：

```bash
/home/wyk/venvs/lutvla/bin/python scripts/run_flashrt.py your_script.py [args...]
/home/wyk/venvs/lutvla/bin/python scripts/check_setup.py --check-import
```

导入入口会移除指向其他 FlashRT checkout 的路径和 editable finder，将配置 root 放在首位，并核验实际 `flash_rt.__file__`。模型加载与后端运行就绪状态尚未验证；工具不会自动安装依赖、编译扩展或启动推理。
