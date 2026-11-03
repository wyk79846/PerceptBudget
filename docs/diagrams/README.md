# 当前机制与 Fast 拟接入方案

2026-11-03。此图描述当前 Full＋Continue 主流程，以及准备增加的 Fast 选择规则；本次仅绘图，没有修改运行逻辑。

![整体机制](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T071408.773361Z-mechanism-flow-e5437335/mechanism.png>)

[矢量图 SVG](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T071408.773361Z-mechanism-flow-e5437335/mechanism.svg>) · [可编辑 Mermaid 源文件](perceptbudget_mechanism_fast_plan.mmd) · [导出脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T071408.773361Z-mechanism-flow-e5437335/render_flow.py>)

- **何时刷新：** 从预测动作寻找夹爪切换、减速、转向，结合剩余动作窗口、完整服务 P95 与 25ms 安全余量，安排下一次刷新。首次刷新和 RTC 窗口约束仍保留。
- **设备配置：** Full 先检查中频能否赶上，赶不上则用高频。无推理的 Continue 执行已有动作，空挡足够长时降至 GPU 306MHz。推理在途时保持计算频率。
- **动作接续：** 后台 RTC 以对齐后的旧动作尾段引导新预测；主循环收到结果后，按实际经过的控制步数去掉新预测的过期头部，接入剩余动作。触发现有关键等待规则时暂停动作，同阶段去重。
- **紫色虚线为计划：** 保持上述 Full 刷新时机与已选频率，只在普通阶段比较同一窗口总能量，允许单次 Fast，下一次必须 Full。推理、调频、Continue 分账，比较时合计。

固定模型配置为 H50、N5、BF16、RTC β5。Full 使用双图共 512 个视觉 token 和 state；本次 Fast 候选使用双图各 160 个视觉 token，省略 state，不训练新模型。Fast 中频成本尚需补测，当前尚未接入正式三频调度。

安全余量 25ms 与频率确认短等的 25ms 是不同参数；关键动作等待没有固定为 25ms。控制名义周期为 50ms，延迟裁剪按实际控制步数，服务预算结合记录的控制间隔。

对应实现：[三频与 Full 选择](../../perceptbudget/three_frequency.py)、[截止时间与关键位置](../../perceptbudget/closed_loop.py)、[RTC 执行与裁剪](../../scripts/probe_delay_buffer.py)。Fast 候选证据见 [双路选择压缩小测](../FAST_DUAL_SELECTIVE_20261103.md)。
