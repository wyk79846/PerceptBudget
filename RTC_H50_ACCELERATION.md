# H50 RTC 推理加速记录（2026-10-20）

按任务书 `PerceptBudget_RTC推理加速任务书_GLM执行版_20261020.md` 推进。
主模型 = pi0.5（H50 checkpoint `lerobot_pi05_libero_h50_quantiles_v044`，FlashRT 编码 + PerceptBudget 侧原始专家）。

## 阶段 1：耗时拆分（已完成）

工具：`scripts/probe_rtc_stage_profile.py`（新增，仅包装计时，不改生产路径）。
固定观测 / 固定 seed 噪声 / delay=12 / executed=15，同一生产 `_predict` 调用序列。
run 目录（均在 `任务产出/perceptbudget/M1/`）：

| run | GEMM policy | 稳态中位 | 主要组成 |
| --- | --- | ---: | --- |
| `20261020T085515Z-rtc-stage-profile-h50-b59fdc02` | heuristic-first | 663.5ms | context 370 + expert 280 |
| `20261020T092801Z-...-b47723dc` / `093216Z-...-494a7f5c` | autotune | 574.7 / 585.2ms | context 280–291 + expert 280 |

分阶段（heuristic-first / autotune）：

| 阶段 | heuristic-first | autotune | 说明 |
| --- | ---: | ---: | --- |
| set_prompt（含分词） | 3.2ms | 3.3–3.4ms | 每请求重新分词+embed+上传 |
| fill_img + img_upload | 1.7ms | 1.8ms | CPU 归一化 + D2D |
| context_replay（GPU） | 370.1ms | 279.8–291.3ms | 视觉 27 层 SigLIP + 18 层 Gemma 编码，CUDA 图 |
| refresh_context | 1.4ms | 1.4ms | KV 拷入专家 |
| step_replay ×5（GPU） | 5×56.1ms | 5×56.1ms | 5 步去噪，每步前向+完整输入 VJP，单步图 |
| tail（CPU 端口） | ~278ms | ~278ms | 与 expert 5×56 重叠（含 `raw.cpu()` 同步） |

结论：
1. **context 编码（~280–370ms）是第一大项**。autotune GEMM 比现行 heuristic-first 快约
   90ms/请求（选核差异），且跨进程结果确定（两次独立进程 raw 逐值相同）。
2. **专家 5 步 = ~280ms**（每步 56ms），第二大项，已在 CUDA 图内。
3. CPU 侧开销（分词 3.2ms、图像 1.8ms、KV 1.4ms）合计 <10ms，非瓶颈。

## 阶段 2a：GEMM policy 对照（第一个优化，已验证）

发现：当前 H50 参考配置把 `native_gemm_policy` 固定为 `heuristic-first`（历史原因是
跨进程选核可复现）。测量显示该 policy 使 context 图慢 ~90ms/请求。

数值影响：同 seed 同观测下，heuristic-first vs autotune 的最终 raw 最大差 0.0093
（raw 域 max 1.38，相对 0.68%；VJP 范数差 <0.3%）。这是 cuBLASLt 不同算法的舍入差异，
不是算法改变。旧模型 H25 时代同一对照差为 0.0069（未过当时 1e-4 等价 gate，故沿用
heuristic-first）。

按任务书口径：策略性变化，需闭环对照后才能接入默认。

## 下一步（未执行）

- 阶段 2b：专家逐步内 fused CUDA kernel（adaRMS 调制、GeGLU 前向+VJP 融合）候选；
  参考已证伪清单（b3-baseline-freeze：A regpipe/shape mapping/MTILE64/stage 专化均证伪，勿重复）。
- 阶段 2c：闭环 task0/q15 真实异步复测（接入候选后）。

## 阶段 2b：专家耗时结构（已完成，重要结论）

同一 665-token 上下文、固定噪声/目标，纯测量不改生产：

| 路径 | 中位 | 说明 |
| --- | ---: | --- |
| 5 步逐步图重放（生产路径，fwd+VJP×5） | 284.4ms | 56.9ms/步 |
| 整段 5 步一张图（新 capture_full） | 284.8ms | **无收益** |
| 纯前向 5 步（一张图，无 VJP） | 106.5ms | parity=0 |
| eager 无梯度前向 5 步（无图） | 543.7ms | Python 开销参照 |

结论：
1. **逐步之间的 Python/启动开销 ≈ 0**（整段图没有变快）。生产专家侧 285ms 里
   **~107ms 是前向、~178ms 是 VJP 反向**。反向占 62%，是专家侧主攻方向。
2. 整段图 capture_full 已实现且 parity=0（raw/VJP 范数逐值同），但没有速度收益，
   仅留作结构验证工具，不接入生产。
3. eager 前向 544ms 慢是因为无图 + 每步重建 DynamicCache 等 Python 开销，
   侧面确认现有逐步图已消掉这层。

VJP 178ms 的构成（待进一步）：18 层 × (adaRMS dx、attention VJP、GeGLU dx、
QKV/RoPE dx) 的 autograd 反向内核。优化候选：
- 用 `rtc_fused_backward.cu`（已有 BF16 输入 VJP 内核：ada_dx/geglu_dx/qkv_rope_dx）
  融合反向小算子——历史在旧模型上试过单层 2–3% 收益，未通过 1e-4 gate；
  这次在 CUDA 图内以逐步 gate 方式重试（差值要求：相对逐步图 <1e-4）。
- 仓库记忆：B3 冻结清单中的 kernel 优化路径已证伪（那是权重侧低比特 kernel，
  与本处激活侧 VJP 不同，不冲突）。

## autotune 决策（2026-10-20 用户指示）

用户决定：**暂不接入 autotune**（保持 heuristic-first 数值口径），待其他优化
完成后再评估是否切换（届时可配合 FlashRT 算法表导出接口锁定跨进程一致性）。

## 阶段 2b-1：融合 GeGLU VJP 候选（已证伪，2026-10-20）

按用户授权尝试修改专家侧运算逻辑（保留完整 VJP/步数/权重/精度，同输入对照）。
实现：`rtc_fused_backward.cu` 新增 tanh-GELU VJP 内核（匹配 gelu_pytorch_tanh），
`rtc_fused_vjp.py` 打包为 torch.autograd.Function 挂进 AST 后缀的 GemmaMLP。
验证（run `20261020T104931Z-...-dd72d64f`，同噪声/旧目标/延迟）：

- 单步 GPU：56.1 → 58.9ms（**更慢**：bf16 下逐元素内核带宽开销 > 省下的启动开销）
- 同输入 raw 差 0.0047 > 1e-4（**parity 不过**：内核以 bf16 舍入 dg/du，autograd
  原生在 FP32 组合；与 2026-10-18 fused FFN 失败同根因）

结论：GeGLU 融合反向在此硬件/精度组合下被证伪，不接入生产。代码保留供复现。
专家侧逐值一致的融合空间剩余：adaRMS dx（同样面临 bf16 舍入关，预期同样证伪）、
attention VJP（历史已证伪 flash 路径）。**bit-exact 约束下专家 ~285ms 已接近下限。**

当前 bit-exact 基线：稳态 663ms = context 370 + 专家 285 + 杂项 ~8。
达到 400ms 需：autotune（−90，用户押后）+ FlashRT context 内核工作（需另行申请）
或放宽逐值约束（需用户另行决定）。

## Codex接手：常量前缀KV梯度裁减（2026-10-20）

没有直接接受“专家已经接近下限”的判断，新增项目侧候选
`perceptbudget/rtc_suffix_vjp.py`，通过原始eager attention前向，只计算动作部分的
dK/dV；常量前缀的梯度最终本就被丢弃。保留对动作输入的完整VJP，不是RTC近似。
候选改变反向GEMM形状，数学等价不代表BF16实现逐值等价，因此必须扩展检查。

证据均在输出根目录的M1下：

| run后缀（完整目录以时间+rtc-stage-profile-h50命名） | 结果 |
|---|---|
| `20261020T110451.148448Z-…-18db4cd3` | 同有效长度665、实际旧目标，3组噪声的5步input/velocity/clean/error/VJP/output全0差；专家284.42→258.78ms，10次交错稳态样本。完整参考请求中位670.69ms，**没有候选完整请求成绩**。 |
| `20261020T110707.084996Z-…-eee5b58a` | 候选接入图准备时，长度662的raw差0.00498、norm差0.00543，门禁拒绝，未开始候选请求计时。 |
| `20261020T110905.234186Z-…-d94dd6ec` | QK分支检查发现长度662不等；后续多图探针触发768MiB保护（实测739.7MiB），exit125，标无效，不算策略失败。 |
| `20261020T111104.407781Z-…-0dfc8a9f` | 加逐图GC/缓存回收及逐项落盘；AV、QK单独在长度662仍分别raw差0.00826/0.00701，均未通过，首个失败即停止该分支；正常退出。 |

**结论：局部确实能省约25.6ms，但不能作为通用保真加速交付。** 尚未定位具体
底层数值内核，不把差异直接归因于某一种cuBLAS算法，也不靠放宽门禁宣称成功。
未进行候选闭环任务；所有成功率及400ms目标均无新增结论。

为保持主线清晰，已撤回本轮试加的正式入口/适配器/图银行开关，这些文件恢复接手时
的内容，FlashRT未改。只保留候选模块、CPU解析对照及原探针的两个显式选项：
`--suffix-vjp-probe`（分支诊断）和 `--requests-only`（跳过其他专家探针）。
原生产默认仍是原始源顺序专家、heuristic-first及完整VJP。

最终CPU回归：371 passed、10 skipped、178 subtests passed；GPU候选检查单独按上述
原始run报告，不能把CPU通过等同候选GPU数值通过。

复现分支诊断（需同机GPU；内存保护和互斥锁保留）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python scripts/guarded_run.py -- \
  /home/wyk/venvs/lutvla/bin/python -u scripts/probe_rtc_stage_profile.py \
  --iterations 5 --warmup 1 --eager-reps 0 --suffix-vjp-probe
```

下一条已知有明显延迟收益的路径仍是此前押后的autotune；需要先获准恢复该实验，
完成数值和闭环质量对照，再决定是否采用，不自动替换。约90ms是既有固定输入
测量，不是保证的新闭环收益；不声称启用后即可达到400ms。

## 专家优先：单独移植快速注意力（2026-10-20）

用户指出先聚焦专家，不动编码器/autotune。回查旧约393ms快版：context约260ms，
专家约123ms，但使用旧模型H20/较小文本容量、自动选核，以及数值不同的专家副本。
RTC_CORRECTNESS_AUDIT.md已定位其FP32动作投影被改为BF16等差异，不能整套恢复。

本轮新增 `rtc_exact_flash_probe.py`，只在隔离的原始专家中把eager注意力换为
Torch Flash SDPA，保留原始context-first KV顺序、精确有效长度、FP32动作投影、
FP32时间与调制、原始去噪/引导及完整动作输入VJP。启动检查mask必须全有效，不
省略非平凡mask；无生产启用入口，退出恢复隔离模块函数。没有改FlashRT/编码器。

证据run：`M1/20261020T111807.189197Z-rtc-stage-profile-h50-6dbd2e1f`。
H50/N5、新模型、有效prefix665、d12/s15、同KV/旧目标，3个固定噪声；双方专家
各10次稳态交错计时：

| 指标 | 原始eager注意力 | 快速注意力候选 |
|---|---:|---:|
| 专家中位 | 287.25ms | 183.88ms |
| 专家P95（仅10样本描述） | 288.98ms | 186.70ms |

专家省103.37ms（36.0%）。三组最终raw相对L2差0.2874%/0.2765%/0.2677%，
最大绝对差0.00505/0.00524/0.00620，不满足逐值/1e-4等价要求。
逐步报告存于 `exact_flash.json`；后续步骤输入已经分叉，不能把其中VJP相对差
当作固定同输入、同残差下的独立梯度误差。

这是数值后端候选，不是当前默认的保真替换；未验证闭环成功率、跨长度稳定性或
候选完整请求延迟。不能直接报告“完整RTC已降100ms”，也不以0.3%数值差推断
成功率安全或不安全。下一步应验证同候选eager/graph一致性及闭环质量，而不是
仅因非逐值相同就认定RTC算法错误。默认原始路径与此前押后的autotune均未切换。

复现：前述guarded_run命令把 `--suffix-vjp-probe` 换成 `--exact-flash-probe`。
本轮CPU全套374 passed、10 skipped、178 subtests passed；GPU测量如上独立报告。

### 首个同任务真实异步验证

用户批准后，显式 `diagnose_rtc_handoff.py --exact-flash-attention` 接入原专家适配器，
默认关闭；bootstrap、图银行、慢回退均用同一候选后端，退出恢复模块函数。
不再把同候选eager/graph门禁冒称原始eager数值等价。每种长度预热时检查全有效mask。

父run `M1/20261020T112222.434191Z-rtc-handoff-comparison-936c5902`；
worker `M1/20261020T112222.834800Z-rtc-diag-actual_history-b5349bd5`。
同task0/H50N5q15/state/beta5/seed7init0，真实异步78步成功、6.2769s，完整请求
稳态606.07ms、固定观测预检559.86ms，14.44Hz，0断供/回退/受控等待，夹爪切换1。
8长度候选自身gate全0，独立202项系统记录审计通过。参考f7c12c37为76步/6.1734s/
713.72ms；初始sim/首帧同，bootstrap state/noise同，raw差.002818。
推理中位快15.1%，但任务耗时未改善，不能从单回合宣布总体质量不退化或节能成功。
推理占空比62.36%→56.44%，尚非控制功耗后的实际节能比例。
源码c893419（元数据启动时afd0ca2+dirty，运行文件此后未改）；375pass10skip178subtests。
详细条件、原始目录和复现命令见父run/结论.md。未自动切全局默认/扩测M1。

### 五任务成对验证完成：建议保留

用户要求两版各5个任务，新增 `scripts/compare_rtc_attention_five.py`，Spatial0–4
每版均新测一次，H50N5q15/state/beta5/seed7init0不变，AB/BA交错、独立worker保护，
每回合系统审计、按结果落盘、可显式续跑。没有复用此前单任务结果。
父run `M1/20261020T113056.566660Z-rtc-attention-paired-five-b1bfff53`，源码b28b89a。

两版均5/5成功，0断供/图回退/异常；原版步数77/111/97/85/123，快速版77/110/96/85/128。
五任务平均步数98.6→99.2；真实任务时间均值7.822→7.776s，基本相当。
各任务完整请求中位数的均值689.70→594.46ms，下降13.8%，不是全部请求的合并中位数。
五对初始sim/首帧/state/noise都相同，2567独立记录检查过。

快速worker额外在初始live KV上做原始eager/Flash同输入bootstrap和guided对照，
噪声与旧目标固定、fork_rng恢复、不影响episode随机数、不计正式延迟；使用7维动作
归一化空间报告。10个样本相对L2差0.131%–0.281%，最大绝对差0.00787。
只覆盖初始数值样本，不声称整条轨迹逐值同或物理空间误差同量纲。

符合本次用户给出的实用保留要求，**保留为后续实验可选加速后端，原版回退仍保留**。
未修改全局默认/FlashRT/autotune，未启动M1正式扫描。下一批采用时显式冻结后端，
同步与异步一致使用，不能混用旧后端同步对照。377pass10skip178subtests。
详细五行结果与复现方式见父run/对照结果.md。

### 约26ms分支的闭环补测

用户要求尝试此前常量KV梯度裁减分支，2617127增加显式diagnostic
`--suffix-only-vjp`，与 `--exact-flash-attention` 互斥；共同初始同输入审计可选
suffix_vjp后端。eager、图银行和慢回退用同一个候选数值后端，保留完整动作VJP，
不是将原始eager的严格等价门禁偷偷放宽。默认均不启用，不影响已保留快速注意力。

父 `M1/20261020T143441.511516Z-rtc-handoff-comparison-5c42eee3`；
worker `M1/20261020T143441.922403Z-rtc-diag-actual_history-a9d9b9d3`。
同新模型/task0/H50N5q15/state32text200/beta5/seed7init0真实异步，77步成功/6.232s，
完整请求687.24ms、固定观测649.43ms，14.62Hz，0断供/回退，205记录审计通过。
初始prefix665上的bootstrap/guided同KV/noise/旧目标7维及32维误差均0，
不能推翻此前prefix662等数值反例或证明所有任务逐值等价。

历史同配置task0原版624b3724为77步/714.55ms，快速注意力458d0ff4为77步/597.53ms；
本次确实少约27ms，但仍慢于快速注意力约90ms。三者初sim/首帧同，原版与本次
bootstrap noise/raw也同。继续优先快速注意力；**目前两实现替换同一attention函数，
不能直接叠加收益**，不声称FlashAttention内部不再存在可优化的prefix梯度成本。
未修改FlashRT/编码器/默认配置，379pass10skip178subtests，未扩5任务或M1。
详细原始目录、比较口径和命令见父run/结论.md。

## 快速注意力继续优化：AdaRMS融合（2026-10-20）

实现提交 `2e5d040`。项目侧新增 rtc_fused_norm.py/.cu，合并归一化前向与输入反向；
保留FP32调制、BF16隐藏层、完整动作输入梯度，未改RTC算法、FlashRT、编码器或autotune。
显式 `--exact-flash-attention --fused-norm --audit-norm-inputs`，默认关闭。

固定输入探针 `M1/20261020T145602.068309Z-rtc-stage-profile-h50-4b4d35be`：
同KV/旧目标，H50/N5/d12/s15，交错测量10次，专家中位179.28→159.60ms，省19.68ms（11%）。
三组噪声的7维归一化动作相对L2差0.251%–0.275%；不是逐值一致。

真实异步父run `M1/20261020T150122.421275Z-rtc-handoff-comparison-fb45d4d8`，
worker `M1/20261020T150122.819161Z-rtc-diag-actual_history-4331ef4e`：
新模型/task0/H50N5q15/state/seed7init0，79步成功，6.320s，完整请求中位591.56ms，
预热547.01ms，实际14.44Hz，0断供/图回退，201项独立记录审计通过。
历史快速注意力task0为77步/6.127s/597.53ms；初始sim、图像、state相同。
完整请求跨run仅观察到约6ms改善，任务时间略增，不能由专家省20ms推断整体省20ms。
初始同输入bootstrap/guided增量7维误差0.142%/0.367%，最大绝对差0.01244。
8长度候选eager/图误差0。独立小张量4档输入幅度（0/.001/1/20），归一化前向与输入VJP
均与Torch参考逐值相同；不代表整个模型所有输入逐值相同。

381 tests passed / 10 skipped / 178 subtests passed；仅一次任务验证，不沿用此前快注意力5/5作为
新融合版本成功率。作为可选候选保留，未切默认、未启动M1扫描。
QKV/FFN矩阵合并无收益且额外占90–378MiB，已从主代码移除；实验快照留在
`M1/20261020T144614.596825Z-rtc-stage-profile-h50-2bc78f5f`。

## 专家继续优化：两项小内核候选未接入（2026-10-20）

范围仅动作专家，FlashRT/图像/上下文/autotune均未修改。基线是已保留的快速注意力加归一化融合，H50/N5，固定KV665、d12/s15，3组噪声、AB/BA交错10次计时。

| 候选 | 基线专家中位 | 候选中位 | 结论 |
|---|---:|---:|---|
| 门控残差乘加融合 | 159.684ms | 158.784ms | 仅省0.900ms，未接正式路径 |
| 归一化warp求和 | 158.376ms | 159.178ms | 慢0.802ms，不保留 |

三组样本的最终动作、每步速度/干净动作/修正梯度和VJP范数均逐值相同；只覆盖这批输入，不构成所有输入证明。
这是专家CUDA图重放探针，不是新LIBERO回合，也未测量新候选完整推理或任务成功率。
第一次探针summary的candidate_kind误标suffix_vjp，应以fused_residual.json和源码快照为准；第二次为warp_norm。
两项候选均已从主代码移除，完整实验源码快照仍在以下目录，可恢复，不污染生产路径：

- `M1/20261020T153906.835356Z-rtc-stage-profile-h50-6e07ce1b/fused_residual.json`
- `M1/20261020T154120.070942Z-rtc-stage-profile-h50-553f0c16/warp_norm.json`

候选阶段全套381pass/10skip/178subtests；回退后主代码恢复到2e5d040实现。未切默认、未启动M1。

## 最终组合：快速注意力 + 归一化融合 + autotune（2026-10-20）

用户决定停止专家优化、启用autotune。9bd157b仅给diagnostic补充
`--gemm-policy {heuristic-first,autotune}`，父/worker均落盘、子进程传递一致；
省略时继承原配置，不改全局旧config。FlashRT源码未改，使用其已有算法筛选接口。

父run `M1/20261020T154440.311596Z-rtc-handoff-comparison-38f581a8`，
worker `M1/20261020T154440.748865Z-rtc-diag-actual_history-fbeb88cd`。
同新模型/task0/H50N5q15/state/seed7init0真实异步：80步成功，6.138s，
完整请求中位492.82ms（上次非autotune591.56ms，跨run约省98.74ms/16.7%），
固定观测预热460.75ms，实际14.88Hz，0断供，8个长度eager/graph门禁误差0，
197系统记录检查通过。算法筛选日志显示实测多个候选；不是只写配置而未启用。
仅单任务，不推断总体成功率，也未单独重测expert分段或宣称专家因autotune提速。
启动源为f19080b+本次CLI改动，9bd157b保存同一实现在途代码；381pass10skip178subtests。

后续使用此组合须显式带三个选项，不能只依赖旧全局配置：

```bash
/home/wyk/venvs/lutvla/bin/python -u scripts/diagnose_rtc_handoff.py \
  --reference-run /home/wyk/workspace/任务产出/perceptbudget/M1/20261020T035931.024165Z-openpi-rtc-actual-cfaef1e3 \
  --conditions actual_history --task-ids 0 --q 15 --horizon 50 \
  --checkpoint-profile h50 --exact-flash-attention --fused-norm --gemm-policy autotune
```

停止继续专家优化，不自动启动M1或扩大样本。后续同步/异步实验均应冻结上述同一后端和autotune策略。

