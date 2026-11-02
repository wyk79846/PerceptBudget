# GPU自动railgate首测：驱动拒绝启用

原始批次：/home/wyk/workspace/任务产出/perceptbudget/M4/20261102T110452.539383Z-gpu-railgate-probe-6ce04d59。

关闭状态2秒检查完成；随后根助手向railgate_enable写入1时，内核返回EINVAL（errno22，Invalid argument）。尚未import torch、未创建CUDA图、未执行任何CUDA恢复试次。没有得到进入/恢复时间或节能量，不能把这次当作“断电不省电”实验。

根助手finally已输出RESTORED，保存原值0；本轮再次读取railgate_enable=0、is_railgated=no、gpu_powered_on=on、force_idle=0，GPU保持供电。没有强制断电或改变频率。

原summary.restored=false是另一个程序错误：父进程等APPLIED期间丢掉了先到达的RESTORED事件，随后EOF又触发异常。修复为独立保留恢复回执；驱动拒绝时输出REJECTED并恢复，主程序记录unavailable及原因，无重复堆栈。5项检查包括驱动EINVAL后自动恢复及父进程保留恢复证据。

原始summary和日志不覆盖，新增postrun_review.json记录恢复复核。

结论：当前驱动/设置不接受通过这个接口启用自动railgate，不能仅凭接口存在就判断可用；尚未确定是平台能力、驱动配置或其他限制。安装的nvgpu模块包含“Railgating is not supported”诊断字符串，但未取得对应内核日志，不能据该字符串断言实际命中该分支。用户态未获内核日志内容，本轮不编造确定原因。

不建议原指令直接重复，也不改用force_idle绕过。若继续定位，下一步只读本次内核报错及匹配版本驱动源码；在确认可用配置之前不接入生产调度。现有Full/Continue调频主线不受本次拒绝影响。
