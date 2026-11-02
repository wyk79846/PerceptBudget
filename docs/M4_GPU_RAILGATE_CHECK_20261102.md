# GPU railgate只读检查

2026-11-02，本轮未更改频率、电源开关、功耗模式，也未加载模型。

平台Jetson Linux R36.4.3、kernel5.15.148-tegra。GPU设备真实路径/sys/devices/platform/bus@0/17000000.gpu，/sys/devices/platform/gpu.0为同设备别名。

|字段|连续三次读取|
|---|---|
|railgate_enable|0|
|railgate_delay|500|
|is_railgated|no|
|gpu_powered_on|on|
|load|0|
|power/control|auto|
|power/runtime_status|active|
|power/runtime_suspended_time|4332，未增加|
|power/autosuspend_delay_ms|−1|
|force_idle|0|

这些读数证明当前采样时刻GPU没有进入railgate，不能证明开启后也无法进入。power/control=auto单独不够，因为GPU驱动自身railgate_enable仍为0。devfreq子设备runtime_status=unsupported不等于实际GPU电源设备不支持休眠；应读上面的GPU本体。

本机/usr/bin/jetson_clocks的do_igpu默认锁频分支（约492—495行）先保存railgate状态，然后向GPU_RAIL_GATE写0。该脚本能解释高性能配置常见的禁用状态，但没有追溯最后写入者，不将其当作本机实际调用历史的证明。当前GPU还锁定1173MHz。

驱动暴露railgate_enable、is_railgated、gpu_powered_on和force_idle等接口，说明有进一步验证的入口；接口存在不等于当前应用/CUDA上下文可以无条件断电。本机nvgpu-ctrl.h约1059行说明确定性GPU channel可能默认阻止railgate；这里的channel不等于torch.use_deterministic_algorithms，不能仅凭训练框架开关推断。

下一步若验证：先保留原值，正常无任务时允许驱动自主railgate，不用force_idle强行关停；以is_railgated与gpu_powered_on实际变化确认，再测试保留CUDA分配/图时能否进入、唤醒后输出是否一致。最后才测试完整模型及唤醒耗时/能量。每轮自动恢复原值；不停止桌面或其他GPU进程，不卸载驱动、不切换功耗模式。

本轮结论：可以继续做可用性小测，不能直接宣布已支持模型常驻GPU断电，也不能把现有低频Continue结果称为railgate。
