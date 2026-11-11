# 外部视角 DivPrune 小测

只替换 Fast 的外部视角压缩：外部256选64，腕部256全部保留，总计320。
Full 仍为双视角512 token。沿用 Full/Fast 交替、原动态刷新和三档频率，
Full 带state、Fast不带state，H50/N5和RTC保持原配置。

筛选依据：[DivPrune，CVPR 2025，算法1](https://arxiv.org/html/2503.02175v2#S3)。
在外部视角经过最终归一化和投影后的特征上计算余弦距离，先选与最近邻
距离最大的token，再依次选离已选集合最远的token。选满64个后恢复原空间顺序。
实际传入LLM的是选中的原始视觉token，没有平均、补零或修改模型权重。

位置处理：

- 外部选中位置仍在0–255，腕部始终为256–511。
- 文本从原位置512开始；专家动作位置使用对应文本长度并补回192的压缩偏移。
- 每次图重放都根据当前画面重算选择和RoPE，逐请求记录64个外部位置。

实现均位于PerceptBudget；FlashRT未修改。新增筛选器
`perceptbudget/external_divprune.py`与`.cu`，通过现有项目侧pipeline子类接入。
`run_m4.py --external-token-selector divprune`启用；默认仍为avgpool。
`--external-selector-pair`在一个常驻模型中比较两种压缩，双方均使用Full/Fast交替。

验证：66项相关测试通过，其中GPU测试对照NumPy算法、原始特征、完整腕部、
RoPE以及改变输入后的CUDA图重放。闭环前也检查实际pipeline的特征和RoPE。

开发小测使用seed27/init1，Object0、Goal3、LIBERO-10任务0，每种方法各一次，
共6回合；不使用seed17/37调参，无额外速度校准。
原始记录：
`/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T085503.250232Z-external-divprune-pair-seed27/`。
历史兼容标签ordinary_full对应平均池化，ordinary_fast对应DivPrune；
具体方法以子回合配置`m3_external_token_selector`为准。

结果（6回合已完成，1次模型加载，全部单回合系统/频率审核通过，0断供）：

|任务|平均池化|外部DivPrune|
|---|---:|---:|
|Object0|280步超时|280步超时|
|Goal3|180步成功|174步成功|
|LIBERO-10任务0|520步超时|319步成功|
|成功数|1/3|2/3|

Fast实际请求中位耗时：高频335.88→343.18ms，中频480.67→486.83ms。
这不是同一输入的微基准；两种路线有不同的后续轨迹和请求数。
含失败的任务＋在途推理总能量为728.93/533.34J；长任务成功提前结束是
总量差异的重要原因，不能把这解释成筛选器单次省电26.8%。

原始值与索引：[results.json](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T085503.250232Z-external-divprune-pair-seed27/results.json)。
实际pipeline检查：特征和RoPE逐元素差均为0；候选17次Fast请求的64个
外部位置均合法，腕部维持256个完整原位置token，动作位置偏移192。

Object0和LIBERO-10任务0的首帧、状态、噪声与首次动作完全相同。
Goal3只有3个RGB通道差1（共393216通道），机械臂state和噪声相同，
首次原始动作最大差0.006572。旧汇总逻辑因此把父运行标为invalid，
但6个回合均已完整结束。该父记录保留；新筛选器探索小测改为记录
首帧差异，不因逐位不同阻断汇总，其他比较入口的规则未变。

固定成本仍是原320-token的初始时延估计，新筛选器的额外计算已经包含
在本次实际请求延迟和整任务能量中。该版本保留为实验开关，未替换原基线。

## seed37长任务10项复测

冻结上述规则，在seed37/init2的LIBERO-10全部10项各测一次，1次模型加载。
成功7/10，超时task0、4、6；零断供，79Full/75Fast。
含失败任务＋在途推理总能量2334.15J，总任务时间192.78s。
Fast实际中位耗时：高频348.30ms、中频482.57ms。
[逐项结果](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T091133.174231Z-external-divprune-long-seed37/results.md)。

用户随后授权保持规则并开启Fast的state，复测同样10项；该开关实验不修改或训练state表示。

## Fast带state：seed37长任务复测

两组均7/10；带state改善task4、退化task3，失败task0/3/6，0断供。
Fast高频中位延迟348.30→441.04ms；中频482.57→613.75ms。
前4项保留，后6项按用户要求关闭重复验证、600MiB保护线一批续完。
[完整逐项报告](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T092334.037016Z-external-divprune-state-long-seed37/results.md)。

## Fast只输入真实8维state

按用户授权，Fast删除24维补零，并将文本计算容量200缩到104；Full仍用32维。
外部64＋腕部256、原视觉位置、RTC、刷新规则和频率规则保持原配置，FlashRT未改。
文本和动作按实际缩短后的prompt对齐；动作专家仍输出原有32维模型动作。

同样seed37/init2的Long10项，本轮8/10成功，仅task0/3超时；32维组7/10。
Fast实际中位耗时：高频441.04→370.09ms，中频613.75→520.29ms，分别减少16.1%和15.2%。
实际Fast文本42–57token，中位50；64次Fast均为8维state，67次Full均32维，0断供。
总任务＋在途推理能量2222.52→2077.37J（−6.53%），包含失败。
task6从超时变成功是总量下降的主要来源；共同成功7项能量反而增加6.48%、
时间增加8.26%，因此只确认本轮单次Fast更快，不将总能量变化全部归因于输入缩短。

首轮加载在首项前触发600MiB保护；内存释放后原配置重启，十项在同一常驻模型中完成，
总计2次加载。正常评测未重复执行setup数值验证；运行已有数据审核通过，最终频率恢复。
没有针对seed37的成败调整阈值或增加例外。
[8维逐项报告](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T104313.196688Z-external-divprune-state8-long-seed37/results.md)。

## 冻结8维版：seed17全40任务

沿用上述配置，seed17/init1四套各10项一次，不按本批结果调参。
Spatial9/10、Object7/10、Goal9/10、Long8/10，合计33/40（82.5%）。
一模型完成，318请求（170Full/148Fast），平均23.77步刷新，0断供、无RAM中止。
Fast实际中位耗时高频361.61ms、中频515.46ms。

全部任务＋在途推理4979.54J、任务时间419.99s；与同seed历史33/40锚点
5064.75J、430.57s相比，能量下降1.68%、时间下降2.46%。共同成功30项
能量下降1.98%、时间下降3.72%。历史锚点为平均池化＋Fast无state，
当前为DivPrune＋8维state，这里是整套方案对比，不把变化单独归因于state。
总成功数不变，Goal增加1、Long减少1；保留原33/40锚点，不自动替换默认。
服务正常退出，频率已恢复；未修改FlashRT。
[40项逐项结果](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T110544.191846Z-external-divprune-state8-seed17-all40/results.md)。

## 稳定阶段允许Fast，关键阶段Full

新增显式入口`--fast-stable`，复用既有feedback_v3模板，与`--fast-alternate`互斥。
仅普通buffer刷新、无临近夹爪切换且不在活动夹爪操作区间时允许Fast；
仍检查原时延/成本门槛，Fast之后至少接一次Full。Full时间表、RTC和频率规则不改。
此入口支持现有`--fast-state-dims 8`及DivPrune原位置输入，不修改FlashRT。

开发seed27/init1三项、1model：Object0成功146步（2Fast），Goal3成功181步
（2Fast），Long0超时520步（4Fast）；合计2/3，29Full/8Fast，0断供。
8次Fast均通过实际记录的普通阶段门槛；关键阶段Full18次，Fast后Full8次，bootstrap3次。
总任务＋在途推理626.656J、任务时间49.805s，含超时，不含准备。
本轮没有同批8维直接交替对照，不据此宣称成功率或节能率提升；保留实验开关，
原33/40锚点和交替版不替换，不自动扩测。未使用seed17调参。
[三任务结果](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T115803.953685Z-stable-fast-state8-dev3-seed27/results.md)。

随后用户要求冻结规则测seed17/init1全40，并因成功率低叫停。完成27项、21成功：
Spatial8/10、Object8/10、Goal5/7，Long未测；13项未完成不计失败。
稳定阶段候选不采用，保留当前8维直接交替33/40版。服务和监控已退出，频率恢复。
[停止记录](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T120357.938532Z-stable-fast-state8-seed17-all40/stop_summary.json)。

## 关键阶段Fast、平稳Full：seed17全40

显式--fast-critical，反转既有feedback_v3阶段选择，Fast之后至少一次Full；
保留Full时间表/RTC/频率，Full32/Fast8、DivPrune64＋腕部256原位置，未按seed17调参。
seed17/init1结果35/40：Spatial9、Object8、Goal10、Long8；较当前直接交替33/40，
新增Object6和Goal3成功，无新增失败。1model、187Full/113Fast、0断供。
全部任务＋在途推理4979.544→4801.159J（−3.58%），时间−3.32%；
共同成功33项能量−1.07%、时间−0.08%。设备已恢复，默认与旧记录均保留。
[结果表](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T123814.968131Z-critical-fast-state8-seed17-all40/results.md)。

## seed47/init1冻结复测：与同seed异步q20对照

80回合全部完成，两组各1model、0断供；只换seed，init仍1，无调参。
当前关键Fast33/40（10/8/9/6），q20为37/40（9/9/10/9）。
任务＋在途推理5286.157/6317.598J，总节能16.33%；共同成功32项省25.91%。
总时间441.102/372.466s，增加18.43%。Long分别6/10、9/10，是主要质量退化来源。
seed17时两组均35/40，总节能27.95%、共同成功33项省25.65%；
可重复观察到共同成功任务的省电，但成功率持平没有复现，不据此自动替换默认。
[新种子完整报告](/home/wyk/workspace/任务产出/perceptbudget/M4/20261106T171521.871944Z-critical-fast-q20-seed47-all40/results.md)。
