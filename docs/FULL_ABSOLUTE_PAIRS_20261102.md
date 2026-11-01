# 高低频Full绝对服务能量配对复核

目的：先复核Full自身的能量差，不用Continue总账解释单次Full差异。复用已有device_service和profile_clock_screen入口，未修改闭环默认、FlashRT、模型或RTC公式。

## 条件

- 一个模型进程，固定Spatial0物理状态、图像、state和旧动作；每个pair使用同一噪声种子，RTC固定delay5/executed5。不是成功率测试，不在服务期间推进仿真。
- 每次都先低频驻留1秒，随后高频恢复后Full，或保持低频Full。驻留不计入服务能量；恢复高频的成本计入。
- 顺序为高5次、低5次、低5次、高5次，共10对、20次Full；后半段反转先后顺序。所有服务输入和raw输出逐对检查，相机不匹配仍重取真实帧，拒绝记录保留。
- 能量直接积分恢复/服务命令至动作包就绪的VDD_IN，包含CPU预处理与提交；不减保持基线，不加Continue，不把准备1秒算成服务。
- 原始功率、实际频率、温度、帧/state匹配、输出差、恢复记录和每对结果完整落盘M4/*-full-absolute-pairs。模型仅加载一次。

## 测量边界

M2原程序明确标记亚秒单次积分为diagnostic，因INA响应尚未完全校准。本次比较10对绝对窗口的均值和原始读数，先检验差异是否复现；重复本身不保证消除传感器边界偏差。不能将这次均值冒充已经校准的纯GPU单事件能量，也不能直接推出闭环总体收益。此限制随结果落盘，不隐去。

40项相关CPU测试通过，含旧入口回归、20次ABBA顺序、同pair种子/输入输出一致检查和直接绝对能量汇总；dry-run通过。尚未启动GPU测试。

## 执行

tmux中执行，sudo仅用于有限时长调频，模型保持普通用户身份：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py \
  --full-service --absolute-full-pairs
```

完成后查看summary.json的absolute_pair_review，核对高/低频平均绝对J、各对差值、频率与输入是否匹配；不使用mean_batch_increment_per_request_j。
