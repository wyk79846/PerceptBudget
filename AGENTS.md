# 协作规则

- 主 agent 分配工作并审查；实施 agent 只写明确分配的文件，不同 agent 不同时修改同一文件。
- 每项任务单独提交。GPU 任务一次只运行一个，并排队执行。
- 实验结果写在项目目录之外；每个 run 保留配置快照、原始证据和结论。阶段未验收前不得称为完成。
- 首轮目标配置候选为当前模型、`H=10`、`N=10`、BF16、`state_enabled=true`；基线状态保持待确认。`N` 是去噪步数，不是动作长度；本轮 q 不超过现场确认的 H，不假设扩展 H，也不复制旧动作或 KV 凑长度。
- 原版公开 `predict(images, prompt, state=None)` 接受可选 state，frontend 会把它编码进 prompt tokens，但不会自动归一化或把 8 维补到 32；不得自行 zero-pad。先从原始 8 维机器人状态追踪 OpenPI/Libero 与 checkpoint 的真实预处理规范。统计文件含 mean/std、q01/q99 而 config 未记录 `discrete_state_input`，不足以判定训练输入规范；未查明则报告缺口、不冻结基线，也不比较多种归一化来挑成功率。smoke 另须确认 state 数据真实进入模型。
- BF16 路径须按设备确认：Orin/RTX 使用 `use_fp8=False`；RTX 必要时使用原版 `FVK_PI05_RTX_FORCE_BF16=1`，并排除 INT8 覆盖。Thor 关闭 FP8 实际走 FP16，不能当作 BF16。
- 不切块、不训练、不复制大型数据或权重。未经明确授权，不改共享依赖、设备功率或频率设置。
- 原版 FlashRT、旧 LUTVLA 和既有 venv 保持不动。尚未实现或未经验证的能力不得写成实验结论。
