# 覆盖功率拖尾的Full路线对照

用户同意重新测量高低频。前一轮0.5/0.68秒短窗结果不再用于推断Full真实能效。本次不改闭环、不加Fast、不修改FlashRT。

## 如何测

复用已有模型与冻结场景准备。一个模型进程、固定Spatial0物理状态/图像/state/旧动作，每对相同噪声，RTC d5/s5。高5次、低5次、低5次、高5次，总20次、10对。

每次先在低频驻留3秒，位于计量窗口外。然后以4秒共同窗口比较：

- 高频路线：开始升频→完整Full→降回低频→保持至共同终点。
- 低频路线：直接完整Full→保持低频至共同终点。

两条路线都从LOW开始，到LOW结束，包含真实升降频，不使用250ms旧准备段拼合；实际切频确认与时间线原样记录。各次返回LOW后至少留3秒记录响应拖尾。若服务/返回LOW耗时超过1秒，4秒窗口不足以保留3秒尾段则标无效，不能截掉尾部继续比较。

VDD_IN约20ms轮询，4秒绝对积分为主指标，不扣保持基线。Full自身耗时、旧短服务积分仍单独保存为诊断项，不把4秒路线能量叫作纯Full计算能量。记录起点前/终点前各0.5秒平均功率，便于检查尾段是否仍未回落；3秒来自M2阶跃跟随观测，不宣称实现硬件绝对校准。

每组5次使用一个有限120秒的原频率控制助手，全部服务共用一个模型；每个4秒测量结束时LOW已读回确认，组末恢复原设备设置的清理在计量窗外。功率、频率和输入/输出逐对核验，失败时保留记录。

## 验证及执行

43项相关CPU检查通过，包含4秒共同边界、3秒尾段、20次路线及HIGH返回LOW过程的虚拟时钟集成测试、长窗能量字段不会误用短窗字段；dry-run通过。尚未运行GPU实验。

在tmux中执行（sudo只控制频率）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py \
  --full-service --full-long-window
```

产物在M4/*-full-long-window-pairs。summary.json的absolute_pair_review是完整路线比较；full_service.json内requests[].route_energy为4秒积分，requests[].energy仍是短服务诊断值，不能混用。原始功率route_power_block*.json、调频route_clocks_*/runtime_clocks.jsonl。先看输入一致、频率匹配、末端功率与完整覆盖，再判断能量方向。
