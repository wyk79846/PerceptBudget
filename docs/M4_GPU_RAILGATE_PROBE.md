> 当前结论（2026-11-02）：本机joint XPU rail启动配置使驱动禁止独立railgate；探针现先只读返回unavailable，不再写开关。详见M4_GPU_RAILGATE_PROBE_RESULT_20261102.md。以下保留初版测试设计。

# GPU自动railgate可用性小测

本轮只用128×128矩阵与一个CUDA图，不加载VLA、不运行LIBERO。CPU/EMC/GPU频率和功耗模式保持原设置；唯一修改为railgate_enable。先检查无CUDA上下文时的自动断电，再保持GPU分配和图比较空闲0.2/1/2秒后能否自动进入、恢复计算。每档关闭/开启各两次，次序反转，共12次计算。

直接读取is_railgated=yes且gpu_powered_on=off才标记确认断电；runtime_status或功率下降不能替代这项证据。tiny CUDA结果必须与128全矩阵一致。主时间为CPU发起graph replay至同步完成，不是首个GPU算子启动，更不是单独的纯唤醒耗时。

记录整个idle＋tiny计算＋2秒尾段的板级能量与20ms状态轮询；采样间隔不是传感器响应分辨率。该实验先回答接口是否可用，不能据短窗口得出完整模型回本时间/节能率；进入时间只报采样首次观察到关断的时间。

权重模型不以root运行。根助手仅接收0/1/STOP、最长180秒、EOF/常规中断自动恢复原值，保留根目录快照；不用force_idle，不卸载驱动、不停止桌面或其他进程。异常时人工恢复命令：

```bash
sudo /usr/bin/python3 scripts/control_gpu_railgate.py --restore
```

普通用户运行（需要sudo密码）：

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/probe_gpu_railgate.py
```

结果自动写任务产出/perceptbudget/M4/*-gpu-railgate-probe，含summary.json、gpu_states.json、power_readings.json和根助手恢复日志。若无CUDA时可断电、图常驻后不可断电，就先定位上下文/驱动约束；不直接换用强制断电。

3项检查覆盖双标志确认、合法单开关写入与根助手异常命令自动恢复；dry-run通过。当前代理执行sudo需密码，尚未做真实设置或GPU测试。
