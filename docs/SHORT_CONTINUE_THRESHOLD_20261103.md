# Continue短空档降频门槛

2026-11-03。原规则要求预计空档至少200ms才进入306MHz。本轮将100ms作为候选，成本表的200ms/1秒历史分类不变。

## 改动

`--idle-min-ms 100`表示：距下一次Full设备准备还有100ms以上时，可以考虑降频。恢复和Full本身的耗时仍由原调度预算承担，不把整个100ms当作纯低频驻留。

新增的100–200ms空档只有在原Full计划已包含变挡确认预算时才降频。原本保持同一计算频率的短空档，降频后会额外需要一个确认控制步，当前先保持原档。200ms以上仍走原规则。Full保持中频优先、高频兜底，不启用Fast或总能量选频实验。

代码：[入口](../scripts/run_m4.py)、[三频控制](../perceptbudget/three_frequency.py)、[常驻配对](../perceptbudget/device_quality.py)。未传新参数时保留200ms，修改前分支`archive/before-short-continue-20261103`。75项相关检查通过，包含实际模型子进程参数转发，以及推理期间不切频、临近启动不降频、原预算保持和配对初态设置。

## 已有数据

原M2的100/150ms静态配对各5对均省电，均值0.219/0.238J；当时使用GPU816/EMC2133，不是当前306MHz档。200ms是成本表代表点，不能据此当作硬件最短降频时间。

当前34条旧运行的实际命令至读回统计：HIGH→306MHz中位17.91ms，306→HIGH17.78ms；中频↔306MHz约9–10ms。控制循环还需要消费确认消息，因此硬件往返不到40ms不等于线上只需留40ms。

## 直接改成100ms的首测

Object3，同初态、同一常驻模型，各一回合。

|门槛|结果|任务时间|任务＋在途推理|Full次数|
|---|---:|---:|---:|---:|
|200ms|157步成功|9.263s|122.207J|7|
|100ms，无新增预算检查|280步超时|15.642s|214.486J|13|

两组初态、首段动作精确相同，零断供，记录/频率/恢复检查通过。100ms组确实多利用了两个短空档，实际低频驻留132/90ms。

前4份预测、前84步实际动作和执行前state完全相同。第51–67步共同动作段，原16.116J、新15.803J，少0.313J；这只是局部窗口，不当作整任务收益。

第68步第二次短降频后，调度增加了恢复确认预算：下一次Full由原第74步高频变成第71步中频。首次实际动作分歧在第84步。不能将失败解释为单纯的模型数值误差，也不能把不同完成结果的总能量差报成节能率。上述新增预算检查用于避免这条明确变化。

[首测与逐次检查](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T085142.852765Z-acceptance-m3-47a8cb4c/threshold_review.json>) · [复算脚本](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T085142.852765Z-acceptance-m3-47a8cb4c/review.py>)。

首次启动曾遗漏新参数的子进程转发，在建图阶段已停止，0回合，不计实验结果；修复后增加入口回归检查。[停止记录](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T084910.624946Z-acceptance-m3-b5c33a33/stop_review.json>)。

## 修正后的配对

Object3再次配对，两回合共用一个模型，采用上述变挡预算检查。

|门槛|结果|任务时间|任务＋在途推理|Full次数|
|---|---:|---:|---:|---:|
|200ms|280步超时|15.576s|204.538J|12|
|100ms，带预算检查|157步成功|9.249s|121.655J|7|

100ms组在第51步利用159.7ms空档，低频实际驻留132.0ms；第68/69步的同档短空档因会增加确认控制步而未降频。该组后续Full在第74步高频启动，避免首测第71步中频的新增预算变化。

两组初态、首段动作精确相同，零断供，1659/923项独立记录检查、实际频率、恢复检查通过，原始功率复算差0。两组第2次后续请求已在44/43步分叉，早于第51步首次新短空档，因此不能将本轮成功差异归因于100ms门槛，也不能用204.538→121.655J宣布整任务节能。恢复后GPU1173/EMC3199/CPU1497.6MHz。

结论：100ms存在局部省电机会，新增预算检查已生效；当前保留为独立候选，默认继续200ms。完整成功率和整任务节能率尚未由这两对小测确定。不继续扩大任务或重测q30。

[逐次检查与分账](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T085841.155667Z-acceptance-m3-140d085d/threshold_review.json>) · [CSV](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T085841.155667Z-acceptance-m3-140d085d/full_frequency_pairs.csv>) · [原始运行](</home/wyk/workspace/任务产出/perceptbudget/M4/20261103T085841.155667Z-acceptance-m3-140d085d/summary.json>)。

两批有效实验各加载一次模型；加上建图时停止的一次，本次工作合计3次加载、4条实际任务记录。代码提交`24f5d6b`，未改FlashRT。

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --three-frequency --start-floor-ms 25 \
  --idle-min-ms 100 --full-frequency-pair --task-indices 4 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

不加`--full-frequency-pair`时只运行候选。50/150ms也可选，但当前306MHz路径尚未闭环验证；先不扩大任务数或重测q30。
