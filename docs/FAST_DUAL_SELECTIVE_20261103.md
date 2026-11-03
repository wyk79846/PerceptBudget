# 双路 Fast 压缩小测

用户决定保留当前衔接方式，继续能耗优化。本轮不修改正式 Full＋Continue、RTC、模型、N5、BF16 或 FlashRT。修改前代码保存在 `archive/before-dual-selective-fast-20261103`。

## 做什么

当前12任务记录中，Full刷新占任务及推理收尾能量56.7%，Continue占43.3%，不含额外3秒测量尾段。因此先尝试降低刷新成本。

|模式|外部视觉token|腕部视觉token|state|
|---|---:|---:|---|
|Full|256|256|启用|
|旧Fast|64|256|省略|
|双路Fast候选|160|160|省略|

候选在每路64个2×2特征块中，选择局部特征较相近的32块合并，其余保持原token。相机及空间顺序保留。分数衡量特征冗余，不代表已经识别目标物体；不训练新模型。完整SigLIP仍计算27层，减少的是后续VLM前缀与专家注意力的实际输入长度。

仅新增实验入口。Full/Continue正式选择和关键动作等待没有改变。

## 固定输入

同一Object task3的30/60/90步观测，固定噪声、RTC旧目标、d/s。一个常驻模型共享权重与专家；探针内同一GEMM形状只选一次算法，避免两种Fast比较时重新调优公共算子。

|模式|单次热推理均值|平均模块输入能量|
|---|---:|---:|
|Full|467.7ms|12.106J|
|旧Fast|320.7ms|7.893J|
|双路Fast|319.7ms|7.873J|

每模式16次，按正反顺序分两块测量；计入输入处理、完整上下文和RTC输入VJP。双路Fast比Full单次刷新能量少35.0%；与旧Fast成本基本相同，目标是改善相同预算下的动作质量。此表没有仿真执行，不能等同整任务节能35%。

三个观测的接下来10步六维动作RMSE均值，旧Fast为0.06122，新候选为0.02856；其中两个观测改善、一个变差。它是动作输入差，不是米，也不是闭环轨迹误差。第30步观测的夹爪切换还出现一步偏移；尾段没有一致改善，不能按均值宣布质量通过。

## 闭环小测

固定q30、同初态seed7/init0、高频1173/3199/1497.6MHz、相同精简采集及RTC。比较全Full与Full/Fast逐次交替；Fast后必须Full。这里检验压缩输入，不接入当前M4动态调度。

三任务一批、一次模型加载、六回合；仿真读取已验证二进制，不在常驻模型期间编译XML。

|任务|Full结果／总能量|Full＋双路Fast结果／总能量|
|---|---|---|
|Spatial0|成功77步／77.19J|成功77步／72.31J|
|Object0|成功144步／118.42J|超时280步／211.23J|
|Goal3|超时300步／238.28J|成功170步／130.81J|

两组各成功2/3，成功任务不同。所有成功、失败和超时能量全部计入：433.89→414.34J，少4.51%；总任务耗时29.592→29.736s，增加0.49%。共同成功的Spatial0少6.33%。这组结果不足以证明相近质量或稳定节能，保留为候选，不替换正式主线。

此前同一Object3的独立一对也保留：两组均超时280步，224.27→212.02J，少5.46%；它不是质量通过证据，不与三任务批次混称同一次模型加载。

三任务的初始图像、state、噪声和首次原始动作均完全相同。六回合系统审计通过、零断供，计算频率和恢复通过，逐条原始功率积分复算最大差1.5e-13J。整任务能量为任务启动至在途推理收尾的VDD_IN总输入能量，不减Continue。

## 代码与验证

- `selective_pool.py/.cu`：双路冗余选择与320-token打包，原416默认路径保留。
- `selective_pipeline.py`、`shared_visual_context.py`：项目侧候选前端，共享权重。
- `rtc_visual_alternation.py`、`audit_rtc_system.py`：候选几何与实际运行记录核对。
- `profile_selective_fast.py`：固定输入及小规模闭环探针，默认不改正式配置。

51项相关CPU检查通过；320/416内核在随机及同分输入下与CPU参考逐元素一致，CUDA图更换输入后仍一致，实际视觉特征打包一致。闭环数据来自真实异步执行，不是固定推理步数。

结果：[热推理CSV](assets/dual_selective_hot_20261103.csv)、[动作差CSV](assets/dual_selective_actions_20261103.csv)、[闭环CSV](assets/dual_selective_rollouts_20261103.csv)。

原始：[固定输入](/home/wyk/workspace/任务产出/perceptbudget/M4/20261103T063036.695970Z-dual-selective-fast-b1d45c97/summary.json)、[三任务闭环](/home/wyk/workspace/任务产出/perceptbudget/M4/20261103T064049.800376Z-dual-selective-fast-2c03f0d2/summary.json)、[独立复核](/home/wyk/workspace/任务产出/perceptbudget/M4/20261103T064049.800376Z-dual-selective-fast-2c03f0d2/postrun_review.json)、[Object3负例](/home/wyk/workspace/任务产出/perceptbudget/M4/20261103T063547.352247Z-dual-selective-fast-8fae3f50/rollout_pair.json)。早期父summary的scope字段仅描述固定输入阶段；实际闭环范围以子回合记录和本复核为准。

复现三任务小测：

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/guarded_run.py -- \
  /home/wyk/venvs/lutvla/bin/python -u scripts/profile_selective_fast.py \
  --source-run /home/wyk/workspace/任务产出/perceptbudget/M4/20261029T100738.715237Z-record-full-1e2d1bc9 \
  --rollout-pair --rollout-only \
  --rollout-tasks libero_spatial:0 libero_object:0 libero_goal:3 \
  --environment-cache /home/wyk/workspace/任务产出/perceptbudget/M4/20261103T052639.079100Z-acceptance-q30-34941cef
```
