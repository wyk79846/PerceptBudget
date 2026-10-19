# 基于7/10慢参考的保真加速（2026-10-19）

## 当前结果

只恢复调度加速，不改原始attention、投影、混合精度、源顺序RTC或仿真。
H25/N5/q5/state80、受控d5均不变。默认慢参考不替换。

| Spatial task2 | 慢参考 | 单步原版CUDA图 |
| --- | ---: | ---: |
| 任务结果 | 成功 | 成功 |
| 完成控制步 | 123 | 123 |
| 完整请求稳态中位 | 1815.74ms | 583.21ms |
| 回合真实耗时 | 43.585s | 15.232s |

完整请求中位降低约67.9%；回合耗时降低约65.1%。图预建8.81秒在任务外，
不能把初始化成本隐藏为不存在。这仍是受控d5，非实际20Hz异步。

基线：`M1/20261019T120123.391569Z-openpi-rtc-actual-e6373b1a`。
候选：`M1/20261019T122945.216798Z-openpi-rtc-actual-bf2821fc`。

## 核对证据与边界

- 25份预测raw、实际noise和previous_raw逐值相同；所有已执行动作、
  全部before场景（含仿真状态）相同。
- 双视角124帧中仅第21帧的1个像素通道相差1级（总48,758,784通道）；
  不能称图像逐位同。该差异下本次模型预测仍逐位一致。
- 24轮实际旧目标与123步来源/裁切审计通过；无断供/重捕获/慢回退。
- 实际prefix566/567/568都在任务前准备；本次没有触发768MiB保护。
- 只完成一个成功任务的成对验证，不把原7/10成功率自动赋给候选。
- 运行时末段释放记录的旧cuda_graph_used字段未包含step bank，后已修正
  为包含step bank；本轮真实后端以请求中original_source_step_graph为准。

## 实现

`rtc_source_step_graph.py`：只捕获一次原始专家forward和原始input VJP。
noise、target、mask、FP32 tau及原始专家时间均为更新的设备输入，
每次依原始源顺序执行5步。没有替换attention或重排KV，没有padding。

为避免state变长时临时捕获，SourceStepBank最多预建8种精确长度、
复用scratch图池，串行重放。未准备长度回退同一源顺序慢参考，
明确记录variable_length_fallback；不能称所有输入都稳定583ms。
`--source-step-profile <慢批次summary.json>`根据同任务历史记录选择准备长度，
仅用于形状预热，不参与动作判断；会核对H/N/state/suite等配置。
新任务/新轨迹仍可能遇到未准备长度，需报告回退比例。

## 专家层门禁

`M1/20261019T122525.634318Z-rtc-conditioning-cache-dbfb73ca`：
精确prefix570/563/557、d/s为12/5、10/10、9/8，五步input/velocity/
clean/error/VJP/output与raw差全0。专家原1456–1458ms，图215–232ms。
首轮beta常量在capture内创建而invalid，失败记录保留；v2将常量提前创建，
不改计算。此门禁不是JAX/XLA或全视觉编码独立等价验证。

另试验固定时间调制缓存（2a52c29b）：三组逐步差0，增加约2.21MiB，
但仅省35–60ms；没有与上述图路径混合或接入默认运行，以免引入额外变量。

## 单任务重现

沿用慢参考命令，额外增加：

```bash
--source-step-profile /home/wyk/workspace/任务产出/perceptbudget/rtc-slow-spatial10-o7jbOe0M/summary.json
```

仍必须显式`--author-order-reference --controlled-delay-five`，不加
`--cuda-graph`或`--experimental-varlen`；后两者属于历史其他路径。
当前不开放此候选的actual-delay组合，先对十任务及回退情况进一步核对。
