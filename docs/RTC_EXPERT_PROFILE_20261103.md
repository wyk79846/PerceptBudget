# RTC专家优化前计时

目标是缩短完整Full服务，为低频Continue留出更多时间，或让更多请求能用中频完成。当前50ms路线及其12任务结果保留，未修改生产推理、FlashRT、模型权重、精度、H50、去噪5步或RTC公式。

## 当前固定输入计时

GPU1173MHz、EMC3199MHz、CPU1497.6MHz，真实保存的双图＋state＋旧动作、d5/s5。有效前缀665，编码容量712。完整请求只含已准备输入的上下文与RTC，不含真实取图、环境执行和队列提交。

|项目|中位耗时|
|---|---:|
|完整请求，3个稳态样本|464.93ms|
|感知/上下文，同3样本|304.85ms|
|专家CUDA图，5个稳态样本|157.72ms|

两种计时分别使用CPU墙钟和CUDA事件，不能当作完整服务P95。专家同输入图/参考动作及修正范数最大差均为0。当前已开启单步CUDA图、时间条件缓存、FlashAttention（包含GQA）、融合归一化和原有感知端autotune。

细粒度GPU活动追踪遇到CUPTI权限限制及追踪器异常，没有得到可信的GPU算子分项表；不将CPU追踪冒充算子GPU耗时。

[计时和门禁](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T103448.148761Z-rtc-expert-profile-d34d29f4/summary.json>) · [计时原始值](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T103448.148761Z-rtc-expert-profile-d34d29f4/timings.json>) · [脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T103448.148761Z-rtc-expert-profile-d34d29f4/probe.py>)。

## 已排除的简单切换

先对实际专家矩阵尺寸做随机固定输入的前向＋动作输入梯度小测，未加载模型。将PyTorch当前cuBLAS偏好切成cuBLASLt后，7种尺寸均变慢；按层数/5步加权的矩阵部分估计74.58→93.78ms。前向和输入梯度逐值相同，但不构成真实模型性能评测，候选未接入。

[矩阵筛选](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T101743.556136Z-rtc-blas-screen-5d9162d7/summary.json>) · [完整脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T101743.556136Z-rtc-blas-screen-5d9162d7/probe.py>)。

## 下一步方向

先用不依赖CUPTI的分段CUDA事件定位专家前向和反向成本，再选单个算子优化。固定旧KV不依赖动作输入，当前注意力将它与动作KV拼接；可以研究在保留动作输入梯度的前提下，是否能省掉最终被丢弃的前缀KV梯度。旧eager裁剪分支与当前快速注意力互斥，不能直接叠加约26ms的历史收益。

加速候选通过后，需重新测量高/中频的完整服务时间，更新调度预算及RTC延迟初值，再比较相同任务的总能量。只缩短专家而沿用旧服务时间表，不保证省下的时间都转成低频Continue。

本次仅完成计时和负向筛选，没有新的加速或节能结论。有效分段测量加载一次模型；此前两个临时诊断脚本问题各导致一次失败加载，共3次模型加载启动。首个脚本名`profile.py`遮蔽标准库，第二个误用了不返回结果的warm接口；均仅修诊断脚本，未升级依赖或修改正式代码，原invalid记录保留。
