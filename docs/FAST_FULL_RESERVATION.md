# Fast提前预留Full服务时间：实验候选

使用--candidate m3_fast --fast-reserve-full启用，默认关闭，与--full-only互斥。

Full/Fast原最晚启动时间仍由截止时刻、对应完整服务分位数、余量和RTC提交上限计算。仅Fast候选额外提前：

reserve_steps = ceil((Full恢复至提交的服务分位数 + eta) / 实测控制周期)

Fast_latest = min(deadline - Q_fast - eta, RTC_latest) - reserve_steps * 控制周期

例如原RTC限制为39步，Full预算加余量为10步，则提前到29步。先应用RTC上限再扣预留，避免RTC先卡到39步后抵消这次提前。Full不扣额外预留；若Fast提前后不可行，选择可行Full。Fast后下一次刷新仍必须Full。低频仅Continue，计算均高频。

每次判定记录原最晚时刻、Full预算、预留步数及秒数；独立审计重算时限与能量、核对预算来自Full历史，不重复算额外能量。该规则预留时间预算，不保证未来没有漏检事件或动作误差；总任务能量与质量须闭环验证。

45项相关CPU测试通过，含39→29、向上取整、Fast不可行退Full、禁止连续Fast、审计拒绝篡改。GPU闭环未运行。建议先Long6一个任务，与现有Full-only成功及原Fast失败比较。
