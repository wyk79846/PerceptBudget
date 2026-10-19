# 原始OpenPI专家RTC参考路径

## 状态（2026-10-19）

已接通独立原始源码路径，固定样本计算门禁通过；真实异步尚未完成验证。
首轮eager完整推理太慢，H20预检判定无法覆盖延迟，未启动正式episode。
追加CUDA图后固定输入完整请求约573ms，见下一节；仍不是实际任务结果。
没有修改FlashRT、环境依赖、权重、硬件频率或项目正式默认。

## 保真CUDA图加速（追加）

新增`perceptbudget/openpi_graph.py`重放原始expert前向/VJP与作者Euler运算顺序。
只将固定FP32时刻与固定suffix掩码张量提前创建；后者仅作用于隔离suffix的局部torch代理，
不改全局torch.tensor。没有缓存/改写time MLP或Ada数值，没有降低精度、步数或改变checkpoint。
图绑定exact prefix shape，KV内容可原地刷新；上下文地址替换时拒绝重放，以防stale KV。
当前CLI强制`--cuda-graph`仅用于`--timing-only`，不允许作为实际episode入口。

|范围|普通执行中位ms|CUDA图中位ms|
|---|---:|---:|
|固定输入原始专家（含RTC梯度）|1449.84|208.83|
|完整FlashRT编码＋原始专家请求|1837.19（此前同配置）|573.29|

专家probe：`M1/20261019T094518.966641Z-openpi-rtc-graph-probe-f275749a`。
五步速度场、五步输入VJP及最终raw与独立作者顺序参考最大差均0；
改变噪声、旧动作目标、d/s参数后，图与普通原始执行最大差0；
刷新KV内容后仍差0，且输出确实改变，未使用旧KV。
单次建图约4.64秒，独立于上述稳态计时。专家不求梯度的普通执行约588.56ms。

完整请求：`M1/20261019T094418.002933Z-openpi-rtc-actual-79c21c1f`。
3次573.29/578.25/568.96ms，中位573.29ms；对比原eager中位1837.19ms约降低68.8%。
实际观测live gate的raw及VJP范数差0，prefix570，建图4.86秒另计。
原raw summary的scope沿用通用async描述不够准确，但timing_only=true/episode_started=false明确；
本轮没有实际控制episode、Hz或成功率，后续脚本已修正timing-only的scope标签。
仍使用heuristic-first，不能直接同历史自写专家393ms的自动选核计时作公平比较。

失败/未采用方案也保留：

- 第一轮578cc0ee捕获被原方法内固定掩码torch.tensor阻挡，invalid；通过局部提前创建常量解决。
- 固定物理KV容量592以支持变长state的55e51f7a：图与同容量原始执行一致，
  但相对有效长度570的独立参考raw差0.008165，未通过1e-4门禁，不用于runtime。
  因此当前只采用exact prefix形状，不用padding换取一个通用图，不在任务中自动重建图。

下一步应解决多个实际prefix长度的保真执行管理，再验证实际异步。
573ms在目标20Hz下仍超过H20可稳定容纳的半horizon预算，不能据此承诺不断供。
315 CPU tests通过、9跳过，另178 subtests通过。所有门禁只涵盖已测输入，不证明全任务正确。

```bash
python scripts/probe_openpi_graph.py
python scripts/eval_rtc_openpi.py --verified-run <passed-gate-run> --task-id 0 --timing-only --cuda-graph
```

## 三个组件

- `perceptbudget/openpi_reference.py`：隔离加载OpenPI自带Gemma源码，直接执行原始
  `embed_suffix`、`denoise_step`和suffix forward方法；只读取原checkpoint动作专家部分。
  不加载第二个完整VLM，不使用项目重写专家、融合反传或预折叠输出投影。
  FP32动作/时间投影与Ada、BF16注意力/FFN，保持OpenPI原始混合精度；权重全部冻结。
  RTC按作者递增时间实现VJP与Euler更新，动作状态/噪声FP32，无训练、无硬前缀覆盖。
- `perceptbudget/rtc_openpi.py`：FlashRT刷新图像/语言/state上下文；转换K的RoPE排列后
  交给原始专家。bootstrap也使用原始专家，不执行原生decoder生成的启动动作。
- `scripts/eval_rtc_openpi.py`：先运行独立固定样本门禁；实际任务必须提供通过的gate路径，
  检查checkpoint/H/N及原始源码SHA256。仅单任务，采用现有算法1延迟队列与实际索引裁切。

AST只提取需要的方法以避开完整OpenPI/JAX依赖与整模型分配，方法体不重写；
原始Gemma模块隔离导入，不覆盖已安装Transformers。参考源码哈希记入每个gate。
这仍共享FlashRT编码器KV，不是整个原始OpenPI视觉模型的独立复现。

## 固定样本门禁

独立参考先以原始OpenPI模块直接计算，再与新源码调用路径比较：

- 五个相同输入下的速度场最大差：均0。
- 五个相同残差下的输入VJP最大差：均0。
- 完整五步RTC动作最大差：0。
- 原始专家所有参数requires_grad=False、grad=None。

通过run：`M1/20261019T092705.748264Z-openpi-rtc-gate-d9835f75`。
独立参考：`M1/20261019T092612.204175Z-openpi-expert-independent-7c7d1e10`。

此前gate a85cb455按旧参考的拆分Euler更新比较，逐步速度/VJP虽0，但完整raw差0.005876；
旧参考用`x-v/N+gain*g/N`，作者源码用`x+dt*(-v+gain*g)`。原始BF16专家对微小FP32输入变化
可分叉，因此统一作者运算顺序后重测，未放宽1e-4门槛。旧失败门禁保留、不用于运行授权。
数学等式相同不等于有限精度下逐位相同；这不证明旧更新顺序解释了全部任务失败。

固定样本通过不是全观测、所有任务或编码器正确性证明，也没有证明论文成功率可直接迁移。

## 实际延迟预检

H20/N5、state开启、文字容量80、两路完整图像、原速动作、s_min5/最近10次delay取max；
仿真优化关闭。为复现数值基线使用heuristic-first选核，不是393ms快版的公平性能对比。

原始专家普通eager执行，完整请求预热后3次：1944.34/1835.93/1837.19ms，中位1837.19ms。
以目标控制周期50ms初始化延迟d=37步，超过H-s=15步。
run `M1/20261019T092808.771622Z-openpi-rtc-actual-571c2e57`标记`infeasible_before_episode`：
未执行正式控制步，没有成功率/断供结果，不当成任务失败，也不冒称完成真实异步验证。
内存保护保持768MiB，进程正常结束，未触发保护；模型加载/建图不在该预热计时内。

下一步应在不改原始专家运算顺序的前提下研究CUDA图等保真执行加速，重新过门禁；
不能通过暂停仿真、裁小延迟、延长旧动作期限或悄悄改H来制造可行结果。

## 命令

```bash
python scripts/audit_openpi_expert.py --author-euler
python scripts/eval_rtc_openpi.py --verify-only --reference-fixture <author-euler参考run>
python scripts/eval_rtc_openpi.py --verified-run <passed-gate-run> --task-id 0
```

GPU任务使用已有guarded_run、GPU锁和独立systemd服务，结果只写任务产出目录。
CPU测试312通过、9跳过，另178 subtests通过。代码引入于3b224fc，后续记录见Git。
# 2026-10-19：state 实际长度切换复核

`probe_openpi_graph.py --changing-lengths`：run
`M1/20261019T095342.025541Z-openpi-rtc-graph-probe-44897537`。
实际 prefix 长度依次 570→563→569→570，不补齐、不截断；每次释放旧图，
按真实长度重新捕获，动作最大差与五步 VJP 范数差均为 0。
原长度五步速度及 VJP 对独立参考也全为 0；专家图重放中位 209.34 ms。

换长度准备耗时（包含原版核对与捕获）为 6.33 / 6.34 / 6.32 秒。
因此本修改仅解决变长时的正确性，不解决连续异步实时性。
适配器明确记录 `context_graph_rebuilt`、`context_prepare_seconds`，
准备时间计入整个请求，不隐藏为稳态推理；仅保留一个长度的图以限制内存。
正式 CLI 的图模式仍限 timing-only，未放开真实任务，也未更新成功率。

同轮旧快版复查：`M1/20261019T095456.833116Z-openpi-expert-independent-1038e871`。
18 层共享 KV 逐位相同后，首步融合/未融合速度最大差均 .016214、
VJP 最大差 .014528；仅恢复原始动作投影后仍为 .012237 / .013376。
未满足 1e-4 门禁，不合入旧融合内核。该检查不是任务成败对照，
也不能据此断言已证明数值偏差导致失败；没有重跑旧失败任务。
