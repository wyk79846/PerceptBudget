# 项目协作与当前状态

## 唯一有效基线

- 以`config.json`及`perceptbudget/baseline.py`为准：新LeRobot LIBERO H50权重、N5、BF16、state启用/容量200、完整VJP RTC真实异步、beta5、source-step图/条件缓存/Flash注意力/融合norm/autotune。
- 正式入口`scripts/run_baseline.py`，默认q5；扫描q5/10/20/30/40。不要恢复旧模型、运动减半、硬前缀或SEAM为默认。
- 参考run和gate路径见配置，现有诊断内部仍读取这些证据。不要删除原始结果；若要解除历史run依赖，作为独立实现变更审核。

## 已完成与待推进

- 最新2026-10-21文档+池化对照：本地docs/deployment_orin.md132明确pool2/4有损需逐任务验证；文档性能120行全pool1，cos0.991属于cache_frames2非池化成功率，bench三preset都pool1且随机双图计时。完整Git引入6091c15未找到当前LIBERO pool2/4质量承诺/强制训练要求。新增诊断scripts/diagnose_pool_mechanism.py只固定输入：一次加载Full模型、复用Full分配，读取失败Spatial2观测5/105/216，同噪声/旧动作/state/延迟，比较Full512、均值复制回原512位置(非加速)、外部均值压缩320、外部原token左上角抽样320，腕部保留。三份手动Full KV与捕获图误差0；动作7维50步RMSE均值压缩=.26062/.09996/.27597，抽样=.17771/.08588/.22614，平均不缩短=.13288/.14873/.14465。前5步误差亦保存；不声称成功率改善/位置是唯一原因，compact改变token数、位置及注意力权重，未实现compact保原位置控制。结果M2/20261021T121757.404855Z-pool-mechanism-501befe4已completed退出0，无后台任务；FlashRT未改/无训练/无新增闭环。报告任务指导与相关参考/PerceptBudget_FlashRT池化说明与固定输入对照_20261021.md；362pass9skip110subtests。备份archive/pre-m2-pool-mechanism-20261021。后续先扩大固定输入覆盖/温和不平均选择，再审阅是否小样本交替闭环，不重启40。

- 最新2026-10-21环境峰值修复：40计划第一组9f2b8c64实际仅task0..6七条，3success4timeout；构造task7外guard761.4MiB125，其他3组未启动，invalid.json补充child running字段过期。旧图释放后allocated6913.4MiB持平，非仍有已证明CUDA流累计。本轮device_quality仅M2进程内在MjSim.free先显式释放旧MjRenderContext缓冲，再调用原free；保留硬重置/EGL不destroy/900与768保护，不改FlashRT或物理精度/功耗。六任务20步gate ae986a81全图像/物理/raw/contact/success差0，44显式free0error；eval_pooled默认gate已更新。诊断脚本补spatial10/resident-source及确定性设置。准备失败错误source/变量遮蔽/遗漏确定性均保留日志，不作通过证据；生产本来deterministic=True。cd3b7606单次模型常驻十个不同Spatial环境创建/校准/硬重置/bootstrap/关闭已completed，含task7，无guard，min记录点1411.60MiB，40free0error，allocated6881.4持平；不是十条闭环。另c9c34dab实际闭环task6→7两条完成退出0，模型专家各1次，task6=108success、task7=220timeout，0underflow/fallback且auditpass，min记录点1206.83MiB，8free0error；不声称40任务已验证/无限长保证。358pass9skip110subtests，无GPU后台运行，不重启40。压缩逻辑核查：未压缩SigLIP特征保留不等于完整独立推理路径，两路在联合Transformer全注意力互相影响；外部pool2还将腕部起点256移至64、文本512移至320，现有连续位置按缩短序列计算非已发现错位；原生均值在finalLN/project前会改特征分布。低成功率更不能靠“一路完整兜底”推断可用，尚无唯一因果证明/不改压缩算法或训练。详细报告任务指导与相关参考/PerceptBudget_M2环境峰值修复与单路压缩核查_20261021.md。备份archive/pre-m2-render-peak-fix-20261021。

- 最新2026-10-21单路池化：用户要求四组各Spatial0..9一次共40，由用户执行。新增项目侧mixed_view_pool.py pipeline子类，在原分配入口前设vision_seq_enc320/272（encoder容量520/472），调用原双路池化后以同流D2D恢复未压缩视角，原顺序外部后腕部，不改FlashRT源码/内核/训练/Full默认。eval_pooled_views.py --pool-factor2 4 --pool-view agent wrist即四组顺序外部2/外部4/腕部2/腕部4；每组独立模型生命周期、组内常驻，不并发40/不逐任务重载。四组固定输入各2请求短测已completed：feafa96d/e243b0c6/c0e31bdb/e1084b6d，410.11/358.99/415.14/356.11ms；CPU池化布局逐元素0（未压缩相机保持原值），RTC动作/VJP范数图对照0，valid473/425=320/272+153，非闭环质量/服务L。357pass9skip110subtests。40任务未启动；preflight已正常结束。备份archive/pre-m2-mixed-pool-20261021。命令仍q5H50N5state开启seed7init0，持续单路简化非Full/Fast交替。

- 最新pool排查2026-10-21：用户因低成功率要求停止pool24服务，已确认inactive/dead、无eval_pooled子进程；pool2批次3204975b完成7条Spatial0..6，task5success其余6timeout，pool4未开始，不重启。原批次stopped_by_user.json说明原running字段过期。新scripts/diagnose_pool_parity.py读取失败task0的观测步5/11/17（valid280/282/283），用同图/state/噪声/旧动作/延迟交替正反序6次，与git b9e7aef^独立KV老图、新共享KV图及无图RTC对照，动作/VJP范数/已保存动作误差全部0；原生池化与CPU逐元素误差0，两视角顺序/前缀valid=128+152/154/155正确。结果M2/20261021T111535.952274Z-pool-memory-parity-d756b3a2/summary.md；诊断退出0，无后台GPU实验。只排除这些记录中的内存优化数值错误；更支持持续75%token缩减的质量/适配问题，不声称唯一原因/否定交替Fast。FlashRT未改，未训练，不继续盲跑。

- 最新双视角pool接入2026-10-21：备份archive/pre-m2-token-pool-20261021，scripts/eval_pooled_views.py --pool-factor2 4顺序两档，每档Spatial0..9各一次/单次模型加载，总20，尚未执行闭环10任务，用户要自己执行命令。H50N5stateq5seed7init0，实际异步完整RTC，持续固定pool而非动态Full/Fast。model.py仅M2显式双视角允许m2_vision_pool_factor2/4；baseline拒绝覆盖；h50_state_prefix_lengths新增pool_factor，visual512→128/32，state tokenizer不变。device_quality新增显式task_plan仅Full新run，复用已有内存修复/共享EGL/900/768保护/audit。固定输入probe e6cd97b2/44896b0a各2请求已completed，324.01/276.55ms，原生encoder容量328/232，valid281/185，动作/VJP范数图对照0，非质量/完整服务L。preflight服务已正常结束，不启动20任务。FlashRT未改/未训练/根配置未改。

- 最新内存排查2026-10-21：备份archive/pre-m2-memory-fix-20261021。rtc_openpi常驻adapter复用建图流；SourceStepBank共享最大KV底层存储，每层仍精确连续前缀，无padding/改模型/改FlashRT。设备小矩阵新流每次留下32MiB，复用流7次无增长；新推理线程短测不增长。仅修流9713997e在第二任务构造环境外guard125（648.6MiB）已停止，invalid.json说明原running字段过期。无模型环境9周期c467d6ab无持续线性增涨。共享KV38abd424三Full task0闭环已正常完成退出0，模型/专家各1次，Spatial78/Object143/Goal122success，0underflow/fallback，3auditpass，51长度动作/VJP范数误差0；释放旧图allocated6916.81/6916.74MiB稳定，首任务185.78MiB节省，记录点minAvailable1186.52MiB，非绝对最小/非无限长保证；无GPU后台实验。用户认为长期单视角要求过严，Fast不限定减相机，下一候选保留两路+vision_pool_factor=2，尚未实现，先固定输入速度/数值后Full/Fast交替质量，不沿用3/3硬门槛否定所有动态Fast。详细见任务指导与相关参考/PerceptBudget_M2连续测试内存排查_20261021.md。

- 最新2026-10-21：Fast校准已完成9/9，所有质量服务已结束，不称仍运行、不重启旧服务。原b689fe6c批次Full task0 Spatial/Object/Goal=77/142/123success（3/3）；agent=Spatial220timeout/Object280timeout/Goal132success（1/3）；wrist=Spatial122success/Object201success/Goal300timeout（2/3）。单视角6回合单会话模型/专家各1次，0underflow/fallback、9auditpass共11456checks，初sim/raw/frame全精确配对；无相机通过预设3/3规则，selection=null，不追加task3留出/不标Fast可用。Full累计3次异常续跑加载，唯一未完成留出Spatial3外guard125不计质量失败，soft reset弃用；共享EGL未彻底解决Full常驻压力，但single本次6回合完成（min avail检查点1032MiB、1EGL/15buffer frees/0errors）。ec2f847改先9条校准，calibration_only=true，full/single completed不代表15/15或完整M2。terminal_status completed，但notify-send失败（Notifications服务启动失败），不能声称已通知桌面/聊天。详细批次summary.md保存原记录与限制；无调频/FlashRT/基线改动。后续可标定Full/Continue，Fast列未通过缺口，动态Fast须另验证。

- 2026-10-21优先状态：前resume服务再次在Full Goal0建图时873MiB触900保护，已停止，不能再称运行；已完成Full Spatial0=77success、Object0=142success均audit/0underflow，Fast尚未开始。继续修复后c12afdb启用仅M2的单线程同GPU共享EGL+释放MjrContext缓冲，保持hard reset/禁止eglDestroy。新gate M2/20261021T101001.012984Z-device-discovery-3a32022b六任务各20步物理/raw/contact/success差0，图像最多3pixel差1；不是soft reset（旧soft gate失败弃用）。339pass9skip110subtests。新服务perceptbudget-m2-fast-quality-shared-egl-20261021已启动，日志/tmp同名.log，续同b689fe6c缺项。profile_device退出保存terminal_status.json并尝试notify-send，桌面通知不等于能自动发聊天消息；不得仅凭启动报告仍运行。此服务终止后读summary及日志更新状态，不自动重启旧服务。原始结果保留，不降低保护线/不逐回合重载/不改变基线或FlashRT。共享EGL是否足以解决长运行尚待实测，不宣称已完全修复。

- 2026-10-21 Fast质量小测进行中：批次M2/20261021T094148.675320Z-fast-quality-b689fe6c，q5，task0三suite×Full/agent/wrist校准；Full提前采task3三suite留出，选定单视角后只测该相机task3，总最多15回合。Full/单视角按几何常驻，agent/wrist共用；不逐回合加载。e5f8dec首Full Spatial0=77success且audit通过；Object建图RAM896MiB触900内部保护，未计任务失败。soft reset对照9292215c状态/像素差异，已拒绝。8ab14f53六任务各20步对照：保留hard reset，仅MjRenderContext析构显式con.free、继续禁用EGL destroy；物理状态/raw/contact/success全同，Goal首帧1pixel差1，其他0，44次释放0errors。da4aef2仅M2 gated启用该资源释放，含resume缺项审计/旧summary备份，337pass9skip110subtests；正式基线与FlashRT不改。续测服务perceptbudget-m2-fast-quality-resume-20261021已启动，日志/tmp同名.log，原fast-quality服务已退出，不重启/重复启动。实时看批次full_quality.json/single_quality.json；已完成回合不重跑。累计Full加载计数包含异常续跑，不称严格总2次加载；900/768保护不降低。未获root调频权限，设备设置仍未改。

- M2续进2026-10-21：用户终端确认EMC rate3199000000/min204000000/max3199000000/locked1，dvfs表含3200000000但超出max不得选；当前权限仍需root，无免密sudo。4ba474a增加隔离m2_profile/m2_view loader与profile_device --fast-probe，不改根配置/FlashRT；545feb3补齐CuBLAS确定性环境。结果M2/20261021T092919.503238Z-device-discovery-e49cfb7b：同已存Spatial0输入+旧目标d5s5，每条件5次，Full/agent/wrist=461.36/326.87/325.97ms，平均请求模块J=12.32/8.51/8.68；视觉512→256、容量712→456、valid665→409，两个形状各自raw/VJP parity0，0fallback。Full和single各加载一次，single复用agent/wrist。只是已准备输入的计算短测，非服务L/闭环成功率/动态节能。首9d4e038c CuBLAS初始化失败无有效样本，保留invalid。12个频率候选及用户EMC来源在run目录，仅提案未调频。下一步Fast候选配对质量验证、受限调频及恢复方案；不宣称已选Fast或三个最终频率组合。

- 2026-10-21已启动M2第一步：新增device_control（只读）、device_profile及profile_device入口；备份分支archive/pre-m2-device-profile-20261021。只读发现当前40W模式、GPU min=max1173MHz，CPU policy0/4 min=max1497.6MHz；EMC debug需要root，沙盒外sudo -n仍要求密码，不擅自改sudo/权限/功耗模式。M2/20261021T091851.698669Z-device-discovery-9018a8cf完成7窗合成matmul忙闲传感器阶跃，20ms轮询、0丢样、值变化间隔中位21.90ms（非硬件更新周期），busy约28.7W/后续idle约10.3W非模型P_on/P_idle。源码num_views=1与序列缩短接口存在，未实测Fast速度/质量，FlashRT和唯一基线未改；未选三个组合、未完成27项小表。继续时先解决EMC读取/控制授权并保持恢复方案，不能宣称M2完成。

- M1主扫描300/300：275成功、25超时。异步q5/10/20/30/40成功28/26/24/27/24，均30任务；零欠载不等于硬实时保证。
- 功率标定26.80896W/9.823655W，模块VDD_IN，非独立GPU/外部完整板级功率；机会估计6.3/13.2/35.7/43.4/47.2%，非已实现节能。
- 两任务阶段候选62.55%unknown，安全B_safe尚不能从H50尾段直接推得。完整结论见归档中的20261021最终报告。
- 用户准备推进M2：三感知档位×三个经筛选的GPU/EMC/CPU频率组合×三起始设备状态。先检查接口、测量边界与切换，再建小表；不提前写M3/M4调度或训练模型。
- 原主扫描及常驻收尾全部完成，禁止重启旧full/graphfix/renderfix服务；不将受控延迟结果当真实异步。

## 工程约束

- 只在主项目`/home/wyk/workspace/PerceptBudget`推进；FlashRT为配置指向的`/home/wyk/workspace/FlashRT-upstream`，不擅自修改。
- 修改FlashRT必须先获授权，并维护其`PERCEPTBUDGET_CHANGES.md`；旧已记录补丁不因清理而撤销。
- 结果仅写配置的外部`output_root`；文档归档在`任务指导与相关参考/PerceptBudget归档/M1相关报告`。历史结果和配置快照不可改写。
- 后续测试优先常驻模型，只有异常才重启；原300任务隔离模式只供历史复现，不用于新批次。保留内存保护、GPU锁、失败记录和退出码。
- 不训练、不更换checkpoint/精度/去噪步数来制造收益。不改任务功耗模式与频率，除非已获对应测试授权。
- 保留正确性参考和必要依赖，不按文件名盲删；旧实验入口删除前做引用检查及Git保存。

## 历史与恢复

- 本轮清理前：Git分支`archive/pre-single-baseline-20261021`。
- SEAM/硬前缀清理前：`archive/pre-m2-cleanup-20261021`。
- 完整历史协作说明与旧README已迁入M1报告归档，旧指令不覆盖当前基线。
