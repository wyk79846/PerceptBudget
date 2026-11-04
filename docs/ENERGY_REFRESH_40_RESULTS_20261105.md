# M4省电优先刷新：40任务结果

平均刷新间隔从原动态方案的12.6步提高到24.1步，总能量下降32.1%，成功36→35项。相对固定异步q30，总能量下降9.5%，成功33→35项。达到20–30步刷新目标；相对q30省电20%的早期期望尚未达到。

|seed17 / init1，四套各10任务|原动态V2|固定异步q30|本轮省电方案|
|---|---:|---:|---:|
|成功|36/40|33/40|35/40|
|整组总能量，含失败|7731.48J|5806.30J|5253.25J|
|平均刷新间隔|12.64步|30步|24.15步|
|平均真实刷新间隔|0.85s|1.71s|1.46s|
|Full次数|541|257|309|
|整组任务时间|442.66s|399.58s|420.62s|
|断供|0|0|0|

本轮各套成功：Spatial9/10、Object9/10、Goal9/10、LIBERO-10 8/10。五个超时为Spatial6、Object5、Goal0、LIBERO-10 0/7，均保留在统计中。与q30共同成功的29项，能量下降6.2%；主表始终统计全部40项。

## 改了什么

- 关键提示先判断是否需要新反馈：新旧预测一致的提示不重复刷新；普通运动提示落在新动作的近期执行窗口内时，继续执行当前计划。窗口以原观测步为锚点，按20步起点加完整服务预算计算，不随每个控制步后移。
- 首段首次明显减速、尚未得到响应的抓放仍可提前刷新；关键步等待已经完成一次响应时，也计入去重，避免因墙钟迟到又重复计算。
- 保留阶段响应记录、转向方向校验和平缓转弯过滤。没有30步上限；H50动作供给、RTC接续与最晚启动检查保留。推理仍中频优先、高频兜底，Continue使用既有低频。

H50、N5、BF16、state200、beta5、checkpoint、完整RTC、请求时双图和任务步数限制均保持基线，FlashRT未修改。项目新增入口为`run_m4.py --acceptance-group m3 --energy-refresh`，搭配原有调频和采集参数；原q30及默认复现入口保留。

## 验证过程

只用seed7/init0的Spatial0/3、Object0、Goal0、LIBERO-10 0开发。第一版平均28.2步，但成功5→3、总能量增加10.0%，未采用。修正版同批原/新均4/5成功，平均17.6→25.6步，Full51→35，总能量844.90→617.22J，下降26.9%；131项相关检查通过后冻结源码`5bdaaad`。

随后完整执行seed17/init1的40任务，一次模型加载，无中断续测。没有根据该组结果修改规则或阈值。新批次44841项系统检查、实际频率/恢复与功率积分核对通过；三个比较批次120回合均重新核对能量积分。设备恢复1173/3199/1497.6MHz。

能量为实测模块输入VDD_IN，覆盖任务执行、推理前的频率恢复与在途推理收尾；不含模型加载、评测准备、任务结束后的频率恢复或单列的终态3秒保持。相邻请求的观测步差用于刷新间隔，不把启动预测或末尾残段算成完整间隔。q30和原动态来自已冻结历史批次，模型进程不同；40项初始物理/state相同，39项初始图像相同，不能当作逐位相同轨迹的单因素实验。

## 节能来源与限制

|相对q30的分账|q30|本轮|
|---|---:|---:|
|Full服务窗口能量，恢复不重复计费|2428.24J|3079.64J|
|Continue窗口能量|3378.06J|2173.61J|

相对原动态，减少无用刷新带来主要收益；相对q30，Full仍多52次，服务能量多651.40J，Continue少1204.45J，净省553.05J。因此这版可以作为后续省电路线，但不能声称已比q30省电20%。这40项只作为验收记录，不用于后续逐任务调参。

## 代码与原始证据

- [刷新判定模块](/home/wyk/workspace/PerceptBudget/perceptbudget/feedback_refresh.py)、[运行入口](/home/wyk/workspace/PerceptBudget/scripts/run_m4.py)、[相关测试](/home/wyk/workspace/PerceptBudget/tests/test_feedback_refresh.py)。冻结分支：`candidate/feedback-energy-frozen-20261105`。
- [seed7开发汇总](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T163604.001521Z-feedback-energy-v2-dev-3c0e3ef0/feedback_metrics.json>)、[seed17冻结计划与源码哈希](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164427.020010Z-feedback-energy-frozen-all40-68cea8c6/plan.json>)。
- [完整比较JSON](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164427.020010Z-feedback-energy-frozen-all40-68cea8c6/summary.json>)、[40任务逐项CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164427.020010Z-feedback-energy-frozen-all40-68cea8c6/comparison.csv>)、[能量复算脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164427.020010Z-feedback-energy-frozen-all40-68cea8c6/review.py>)。
- [本轮原始运行及逐任务记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164624.062759Z-acceptance-m3-5a8a5238/recording.json>)、[频率恢复证据](</home/wyk/workspace/任务产出/perceptbudget/M4/20261104T164427.020010Z-feedback-energy-frozen-all40-68cea8c6/clock_after.json>)。
