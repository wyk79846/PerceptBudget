# GPU1173与306MHz切换耗时

已有驻留模型低频试验中，306MHz档下频/上频设置与读回中位约17.01/17.54ms；但原助手同时写GPU、EMC、CPU，不能解释为GPU单独切换时间。

新增scripts/probe_gpu_clock_switch.py：准备阶段固定EMC2133MHz、CPU1497.6MHz，随后仅写GPU上下限，在1173/306MHz间往返10次，每次读回当前频率后保持200ms。记录命令至设置读回、命令至实际cur_freq首次匹配的时间及中位/P95/最大值。P95采用10样本最近秩，因此等于该轮最大值。每次验证非GPU设置未变化。

不加载模型、不导入CUDA；结果只说明当前驱动接口的静态升降频耗时，不是GPU断电唤醒、首算子启动或完整推理延迟。使用既有40W检查、根助手互斥快照和finally恢复。准备与结束可以恢复EMC/CPU设置，但计时切换期间不写它们。正式生产配置不变；低频闭环接入另行验证。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/probe_gpu_clock_switch.py
```

普通用户启动，sudo仅用于有限时钟助手；全程通常数秒，不需要重复加载模型。输出M4/*-gpu-clock-switch-*/summary.json及clocks.log。异常时先核对clocks.log中的RESTORED；若确有未恢复快照，使用原control_device_clocks.py restore。

检查：GPU-only写入、非GPU变更拦截、统计完整性及原时钟控制共17项测试通过；dry-run通过。设备测量等待用户sudo授权，尚无新结果。
