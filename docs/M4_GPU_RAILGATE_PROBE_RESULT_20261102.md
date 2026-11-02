# GPU自动railgate首测：驱动拒绝启用

原始批次：/home/wyk/workspace/任务产出/perceptbudget/M4/20261102T110452.539383Z-gpu-railgate-probe-6ce04d59。

关闭状态2秒检查完成；随后根助手向railgate_enable写入1时，内核返回EINVAL（errno22，Invalid argument）。尚未import torch、未创建CUDA图、未执行任何CUDA恢复试次。没有得到进入/恢复时间或节能量，不能把这次当作“断电不省电”实验。

根助手finally已输出RESTORED，保存原值0；本轮再次读取railgate_enable=0、is_railgated=no、gpu_powered_on=on、force_idle=0，GPU保持供电。没有强制断电或改变频率。

原summary.restored=false是另一个程序错误：父进程等APPLIED期间丢掉了先到达的RESTORED事件，随后EOF又触发异常。修复为独立保留恢复回执；驱动拒绝时输出REJECTED并恢复，主程序记录unavailable及原因，无重复堆栈。5项检查包括驱动EINVAL后自动恢复及父进程保留恢复证据。

原始summary和日志不覆盖，新增postrun_review.json记录恢复复核。

结论：当前驱动/设置不接受通过这个接口启用自动railgate，不能仅凭接口存在就判断可用；尚未确定是平台能力、驱动配置或其他限制。安装的nvgpu模块包含“Railgating is not supported”诊断字符串，但未取得对应内核日志，不能据该字符串断言实际命中该分支。用户态未获内核日志内容，本轮不编造确定原因。

不建议原指令直接重复，也不改用force_idle绕过。若继续定位，下一步只读本次内核报错及匹配版本驱动源码；在确认可用配置之前不接入生产调度。现有Full/Continue调频主线不受本次拒绝影响。

## 继续检查：定位到共用电源轨配置

本机只读证据：

- model=NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super；compatible包含p3768-0000+p3767-0000-super/tegra234。
- /proc/device-tree/chosen/nvidia,tegra-joint_xpu_rail存在，长度0；设备树布尔属性存在即true，不能因内容为空当成false。
- 同一属性在/sys/firmware/devicetree/base/chosen也存在。
- 安装的nvgpu.ko包含nvidia,tegra-joint_xpu_rail、XPU rails are joint、Railgating is not supported三个对应标记；模块sha256=b6fa64d265c46da654c3e8a95eade6e5a79df5d987ed04098597e200656c4f2e。

NVIDIA rel36驱动源码（官方代码的OE4T镜像）：

1. [platform_ga10b_tegra.c](https://github.com/OE4T/linux-nvgpu/blob/patches-rel-36/drivers/gpu/nvgpu/os/linux/platform_ga10b_tegra.c)，ga10b_tegra_probe读取chosen中的joint_xpu_rail；为true时设置platform->can_railgate_init=false，并清除NVGPU_CAN_RAILGATE。
2. [sysfs.c](https://github.com/OE4T/linux-nvgpu/blob/patches-rel-36/drivers/gpu/nvgpu/os/linux/sysfs.c)，railgate_enable_store在输入值有效且请求改变状态后，检查can_railgate_init；false时输出Railgating is not supported并返回−EINVAL。同值写入仍成功，因此“原值0可以保留、写1失败”与实测一致。

原因：本机启动配置声明XPU共用电源轨，驱动据此禁止GPU独立railgate。不是CUDA常驻阻止，因为失败前没有创建CUDA上下文；也不是普通文件写入权限，因为是根助手收到驱动EINVAL。

边界：这里核对的是rel36同系列源码与本机设备树/模块，不冒充已获得r36.4.3逐字源码及运行时内核堆栈。内核日志和lpwr_enable读取仍需终端sudo密码，尚未取得；没有将其虚构为已确认日志内容。当前证据已经给出直接、匹配的拒绝链条。

修改：探针在上述布尔属性存在时先只读返回unavailable，不再索要sudo或写开关。6项检查通过。实际执行新预检记录：M4/20261102T111449.406887Z-gpu-railgate-preflight-4ef45b0b，GPU仍on、force_idle未使用，未改变设备状态。

项目结论：当前设备/驱动配置将GPU独立断电候选标为不可用；不将其计入节能空间。按Word“可用的railgate”要求如实交付即可，继续使用已验证的降频等待。不修改设备树、驱动或功耗模式来绕开该硬件配置判断。
