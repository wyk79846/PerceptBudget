# LeRobot RTC 独立对照（2026-10-20）

## 环境与边界

- 独立环境：`/home/wyk/venvs/perceptbudget-lerobot`，新增约437MiB。
- 通过该环境的`00-existing-gpu.pth`只读复用lutvla site-packages中的Jetson Torch2.7；
  新依赖安装在新环境自身，原环境Torch2.7/Transformers4.53.2未改。
- 原本地LeRobot HEAD为146d847，要求Python3.12。本次固定官方v0.4.4：
  `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`，Python3.10；上游工作树干净。
- 使用官方pi extra指定的Transformers fork精确提交
  `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`（4.53.3），只装入新环境。
- 安装了推理导入所需的draccus0.10、torchvision0.22、datasets4.8.5等；
  完整freeze在任务run中。未配置所有真机/可视化/训练extras，
  现存cmake/wandb等非推理依赖仍有版本约束警告，不宣称全量pip check通过。
- `scripts/eval_lerobot_rtc.py`直接调用未修改的LeRobot PI05Policy、RTCProcessor、
  ActionQueue及LatencyTracker；只增加LIBERO观测、控制循环和低内存加载适配。
  **不是官方一键LIBERO异步评测入口，也不是完全独立的官方处理器管线复现**。
- 同checkpoint、Spatial task2、seed7/init0、H25/N5/q5、双图、state8/token80。
  使用原实验的旋转/缩放、state归一化与clip后动作反归一化；官方文字处理器
  用max_state_dim8而非默认32，本地同一SentencePiece词表。
- 官方输出7维，剩余动作输入RTC时其余维度由官方实现补零；原FlashRT参考保留raw32。
  完整视觉编码换为LeRobot原生，故速度和算法不是严格单因素对照。
- 权重meta初始化、逐张量按官方目标dtype转CUDA、813项strict加载，恢复声明的
  embedding/lm_head共享权重和3个非持久buffer；没有容忍缺失键、随机权重或训练。
  未修改FlashRT或LeRobot的模型/去噪/RTC/队列源码。

## 已完成结果

预检`M1/20261020T031733.931978Z-lerobot-rtc-5254a8ed`：
严格加载成功；无旧前缀推理2.019秒、带RTC推理1.224秒。只两次调用，非性能统计。

实际任务`M1/20261020T031915.307773Z-lerobot-rtc-a16ac148`：

- 220步timeout，控制阶段14.092秒（不含加载、预检、任务bootstrap及末尾worker等待）。
- 8个已提交后台请求，推理中位1543.28ms；实际控制频率15.61Hz。
- 官方示例使用`ceil(wall_time / 0.05)`更新延迟并裁切，得到32–33步，超过H25，
  每次提交剩余动作均为0。第一轮从第5步到第30步，旧队列只剩20动作，
  先耗尽，且新预测被裁空；从第25步起至219步共195步断供。
- 无动作时适配层发送零六维运动＋保持上次夹爪命令，让仿真继续走，不暂停等推理；
  这不是继续执行无限旧chunk。控制队列物理动作放在CPU，避免`.cpu()`等待
  后台CUDA默认流而将控制串行化。每次真实消耗和名义延迟均保存。
- 没有触发768MiB保护或异常退出。完整request/control/frame及源码hash均已保存。

**不能由本次timeout判断LeRobot抓取成功率，也不能用14.09秒声称比原实验快**：
绝大部分控制步在断供保持。该组合先受原生推理速度和预测覆盖长度限制，
还没有形成可用于判断RTC动作质量的有效对照。

后续若扩大H或改变延迟时基，必须单独标注配置变化并得到用户确认；不自动扩测。

## 入口

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python scripts/guarded_run.py -- \
  /home/wyk/venvs/perceptbudget-lerobot/bin/python -u scripts/eval_lerobot_rtc.py
```

添加`--preflight-only`只验证加载和两次推理。此入口默认重现上述H25限制，
不能把它包装为已解决断供的正式实验。

## 固定5步质量对照入口（待用户执行）

新增`--controlled-delay-five`：后台推理时只执行5个旧动作，然后仿真等待，
即使预测提前完成也不提前提交；固定d=5，提交时仅裁5步。仍由原ActionQueue
完成替换，并检查实际消耗恰好5步；默认真实延迟入口保持不变。

为消除前述7维/32维旧目标的已知不一致，此模式仅将LeRobot配置的output feature
设为32（模型本来输出32维），传完整raw32给RTC，执行仍只取前7维。
没有修改上游RTC公式、模型或权重。记录bootstrap、每轮noise/previous/raw/token
及控制/图像，供后续固定输入核对；闭环成败差异本身不等于算法错误。
等待策略两项CPU单测通过，尚未执行该模式GPU回合，不声称已成功。

## 用户执行的固定d5与后续同KV核对

用户run `M1/20261020T032725.551115Z-lerobot-rtc-54558bf7`：160步success，
38.732秒、推理中位1263.02ms，0断供；30次已提交请求的预测延迟/实际消费/裁切
均为5。我们的缓存图参考5384eecc为124步success/14.589秒/551.49ms。
初始画面、完整before场景和bootstrap噪声全同，但尚未RTC的bootstrap raw
已最大差0.00374055；不能仅按闭环差异判断RTC实现。

为进一步排除encoder，`scripts/audit_rtc_lerobot.py`在**同一个冻结原始专家**上
向两套RTC提供相同KV、noise、previous，H25/N5/d5/s5；只调用LeRobot原模块，
没有调用FlashRT encoder，也没有修改上游源码。三组有效长度570/563/557。
最终证据：`M1/20261020T033514.839857Z-rtc-lerobot-fixed-kv-c620bb7f`；
初轮533ffccf也保留。

- 每个同输入去噪步的模型velocity差全0；mask最大差约5.96e-8，
  clean/error差约1e-7级，属于次要浮点运算顺序差异。
- LeRobot修正量与它自己的weighted error逐值同；与作者完整VJP修正
  最大差约0.399–0.402。完整五步raw最大差0.24626/0.24241/0.24148。
  **因此不是仅FlashRT算子选择或BF16舍入差异。**
- 此官方v0.4.4的`RTCProcessor.denoise_step`先调用速度场，再执行
  `x_t.requires_grad_(True)`，速度场计算没有输入梯度路径；之后对
  `x_t - time*v_t`求输入梯度，只保留identity项。因此该版本的表现
  等价于忽略速度场输入雅可比，而非作者完整输入VJP。
- `test_lerobot_guidance_behavior.py`以v(x)=0.1*x²、time=0.5做CPU解析对照，
  完整修正应为error*(1-0.1*x)，原模块返回error；CPU小例子通过。
  联同控制等待测试在独立环境3项通过。该测试是行为刻画，不是替换算法。
- 同时执行作者原函数体经既有窄Torch桥接器的对照（非JAX/XLA）：三组
  每步input/velocity/clean/error/VJP/output及最终raw与我们的源顺序参考差全0。
  证明边界是这三组共享专家/KV输入，不是独立官方encoder或所有任务等价证明。

结论：不能把当前LeRobot版本当作与作者完整RTC数值等价的标准；
也不应仅因LeRobot这一次成功便把项目的完整梯度改成其行为。
此结果只定位去噪引导差异，**不证明项目真实异步调度无问题**，
也不能由单次闭环结果判断完整VJP或identity修正谁成功率更高。
本轮只核对，没有修改任何一边的推理、调度或上游算法。
