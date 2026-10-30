# 常驻 Full/Fast 图缓存内存修复

2026-10-30。旧版任务切换会丢弃并重新捕获全部 Full/Fast 图，新版每个几何族保持有界常驻缓存：保留相同精确长度，仅捕获缺失长度。Full最多17张、无state Fast当前1张；淘汰时同步设备并显式reset CUDA图，再释放张量引用。两族复用同一capture stream和scratch pool。容量712仅用于KV存储，attention仍按真实长度切片，不增加输入token。

模型每批加载一次。900MiB保护线保持不变。未修改FlashRT、权重、RTC公式或任务动作。

## 验证

- 94项相关CPU测试通过，含复用、淘汰和图释放顺序测试。
- 冻结专家、合成KV五轮长度切换：共15次Full/Fast/Full与参考比较，最大raw差0；PyTorch allocated五轮均1218.003MiB，RSS约1542MiB。见 `/home/wyk/workspace/任务产出/perceptbudget/M4/20261030T113609.722183Z-resident-dual-memory-check-ba9b3bc9/summary.json`。
- 无主模型、九次真实环境创建/校准/关闭：完成，关闭后系统可用内存未呈持续下降。见 `/home/wyk/workspace/任务产出/perceptbudget/M2/20261030T113955.816947Z-environment-memory-fc17e8a1/summary.json`。
- 完整模型仅加载一次，真实Spatial0/Object0/Goal0环境连续准备并各回放Full/Fast/Full：全部完成，Full重复输出raw差0，最低系统可用内存1268.7MiB。每任务新捕获图数量分别18、8、5，而非每次18张。见 `/home/wyk/workspace/任务产出/perceptbudget/M4/20261030T114456.772106Z-resident-full-memory-check-c44e5683/summary.json`；探针源码随产出保存为probe.py。

上述为内存与计算一致性验证，不是任务成功率测量；尚未完成新版调度的长任务闭环或12任务运行，不能据此声称长期内存问题彻底排除。
