# 容量200→64为何改变轨迹：诊断结果

## 后续：80容量固定选核验证（2026-10-19）

项目侧新增可选 `native_gemm_policy=heuristic-first`：代理原生GemmRunner，
将每次BF16选核的候选数固定为1；FlashRT源码不变，默认仍为autotune。
这不是保存此前最快算法，而是建立新的首候选数值基线，不保证跨设备/版本一致。

- 固定同图像/state/noise/target，80a/80b两次独立进程的所有已记录阶段逐位一致，
  包括encoder、RTC raw/VJP和native bootstrap；两侧padding污染检查均通过。
- 结果：`M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047`。
- 首候选80计时：完整RTC中位477.01ms，普通推理392.48ms；context345.86ms，
  RTC专家123.15ms。固定输入各5次，不是任务成功率或真实20Hz异步结果。
  结果：`M1/20261019T055014.598798Z-rtc-vs-native-timing-f31a4585`。
- 同容量原自动选核：完整RTC中位392.87ms（5次390.34–394.47ms），普通推理304.33ms，
  context259.97ms、RTC专家122.79ms。
  结果：`M1/20261019T055156.977668Z-rtc-vs-native-timing-1332fc02`。
  相较首候选快84.14ms，但不是逐位等价的同后端替换，也未验证跨启动自动选核的一致性。
- 首候选有性能代价，不作为加速默认；要同时保留速度和固定数值路径，后续应保存并重放
  已调优算法，而不是固定选第一个。现有原生绑定未提供算法导入/导出接口，本轮未扩改FlashRT。
- CPU测试304通过/9跳过（另178 subtests）；未训练、未新增任务评测。

诊断命令：`scripts/diagnose_state_capacity.py --repeat80 --gemm-policy heuristic-first --fixture-from <历史fixture目录>`。
计时命令：`scripts/probe_rtc_vs_native.py --optimized --state-prompt-capacity 80 --gemm-policy heuristic-first`。

日期：2026-10-19。仅诊断，未修改FlashRT、模型权重、生产推理逻辑或默认配置。

## 结论

不能把变化笼统归为“减少空位必然造成BF16舍入”。本次发现了两类底层数值路径变化：编码器的启动自动选核，以及原生bootstrap注意力在不同物理容量下的计算路径。未发现语言/state截断、位置偏移或padding泄漏。

此前“新后端参考核对通过”只验证该后端内部RTC公式一致，不代表不同容量、不同启动选核的完整模型逐位一致。固定seed和Torch确定性开关不锁定FlashRT原生cuBLASLt的自动选核。

## 固定条件与检查

同一份图像、归一化8D state、语言、初始动作噪声和旧动作目标；H20/N5、同一checkpoint和FlashRT提交。依次在独立进程跑capacity200a/64/200b，保留逐层BF16位模式。诊断追踪的eager有效KV与各自预热的原计算图逐位相同，避免把追踪本身的差异误当容量差异。

三个worker均确认：

- 58个有效文字token，512个图像token，有效prefix=570；分词、有效语言embedding、27层视觉输出、encoder/decoder RoPE一致。
- encoder有效长度570、decoder有效长度590、decoder起始位置570，一致。
- 把encoder屏蔽文字行填为100，有效KV不变。
- 把RTC屏蔽KV改为K=11/V=-13，RTC输出不变。
- 权重冻结、无权重梯度；没有执行新任务或改变动作调度。

## 发现一：同容量跨启动的自动选核也会改变结果

首轮200/64/200对照，有效encoder各层及固定输入RTC输出全部逐位相同。

第二轮64容量的自动调优，对encoder FFN down形状选择了另一候选算法（启动日志对应条目best=1变为best=5）。这两次64运行第一次差异均定位到encoder第0层的FFN down GEMM输出，进入该GEMM的输入逐位相同；200容量跨两轮与重复对照仍逐位相同。第二轮capacity200/64对照也从该处首次分叉。

该隐藏层输出相对L2差约0.1945%，最大绝对差2.0，约24.1%元素不同；这是编码器隐藏张量，不是机械臂动作或米。最终固定输入RTC raw相对差约0.7917%、最大绝对差0.013496。第一轮没有这些差异，因此容量变化本身不足以解释全部变化，启动自动选核是必须控制的混杂因素。

源码依据：FlashRT `pipeline_rtx.py::autotune_gemms`对各M/N/K形状调用`autotune_bf16_nn`；`gemm_runner.cu::autotune_cached`按实测速度在候选cuBLASLt算法中选最快并缓存。这里不进一步声称是某一种内部精度/reduction scheme，未直接读取算法内部配置。

## 发现二：原生bootstrap首块已能复现历史差异

旧任务step0不是RTC可微专家生成，而是普通FlashRT native decoder。用相同Torch seed7噪声复现后：

- 200与64的噪声逐位相同，初始投影、首层QKV、动作K/V及encoder第0层prefix K/V也相同。
- 首个native注意力输出开始不同（max0.015625，相对L2约0.0217%）；有效长度一样，但物理KV长度分别732/596，原生split-KV路径随物理形状配置计算/归并。
- 完整首块raw最大绝对差 **0.0078125**，相对L2约0.4034%。两侧各自逐位匹配此前104步与187步任务日志中的首块，确认历史轨迹在启动动作阶段已经分叉。
- 仅把native decoder改为无split的诊断对照，完整首块仍不同，因为上游encoder也存在选核差异，不能把全部原因只归到split-KV。

“首块起点已定位”不等于已证明每一段轨迹/成功步数均只由此造成；但不能再把这些任务时间差直接作为纯计算优化收益。

## 数据与复现

- 首轮：`M1/20261019T053108.377723Z-capacity-numeric-diagnosis-aeab67f7`。
- 追加bootstrap追踪：`M1/20261019T053555.980538Z-capacity-numeric-diagnosis-c81702d8`。
- 首次探针导入OpenPI失败：`M1/20261019T052952.475336Z-capacity-numeric-diagnosis-24c1f1e1`，标记invalid，未计入结论；修正探针为实际生效的SentencePiece回退路径后重试。
- 入口：`scripts/diagnose_state_capacity.py`；`--fixture-from <前轮run>`复用相同fixture。每worker通过guarded_run及GPU锁，原始trace.npz、自动调优日志、首差异表和屏蔽检查保留。

下一步如果要做严格速度对照，应先固定并记录原生GEMM选核和bootstrap数值路径。该修改尚未实施，不自动调整推理或重测成功率。
