# 原始OpenPI专家RTC参考路径

## 状态（2026-10-19）

已接通独立原始源码路径，固定样本计算门禁通过；真实异步尚未完成验证。
原路径完整推理太慢，H20预检判定无法覆盖延迟，未启动正式episode。
没有修改FlashRT、环境依赖、权重、硬件频率或项目正式默认。

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
