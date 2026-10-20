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

