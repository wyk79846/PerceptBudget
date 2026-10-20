# 协作规则

- 用户指定β10，显式CLI增加10（默认仍5）/τ0测试增加10，15定向CPU测试通过。父run M1/20261020T061148.961341Z-rtc-handoff-comparison-92795af5，workerfb7f8806 task0H20N5q5state80固定d8仍220timeout/17.262s/median581.73ms，夹爪指令切12次/碗maxlift0.0077cm，0断供/回退；8形状gate通过/845独立记录检查通过，26后台实际d8。beta10全部请求生效，bootstrap与β5相同、首guided噪声相同/raw差0.00224343。beta2/5/10本任务各一次均超时，不能据此宣布全局最优或公式错误，不自动追加，默认5保留；详细父run/结论.md。

- β2试验已完成：父run M1/20261020T060623.787243Z-rtc-handoff-comparison-fe3a4be3，worker6b5a419f，task0/H20N5q5state80固定d8/估计8仍220timeout/17.322s/median583.42ms，夹爪指令切14次、碗maxlift0.22cm；β5对照切8次/maxlift1.11cm，均未抓稳完成。0断供/图回退，26已提交d均8，845独立检查通过；8长度beta2图/eager输出与VJP全0；所有requests beta2，bootstrap相同，首guided noise相同/raw差0.0480076，确认参数真实生效。未自动追加β1/更多任务，默认仍5，不因两档单次失败宣称公式错。详细父run/结论.md。

- 用户批准同task0固定d8降低β试验，本轮只试β2（H20N5q5state80seed7init0其余不变），不自行扩β1/十任务。原source-step beta写死5，新增可选rtc_beta/diagnostic --beta，默认5不变，step图与eager回退/准备gate均传同β并记录request；legacy完整图禁止非5。14项CPU定向测试通过（含τ0 beta1/2/5逐值对参考及非法值），GPU现场各长度beta2参考/图gate通过后才可执行任务。FlashRT/模型/调度未改，无训练/子agent。

- 用户允许同task0固定d5补充，父run M1/20261020T054745.609999Z-rtc-handoff-comparison-deca92a5，workerbcf86118：198步success/23.207s/median566.37ms/夹爪指令切5次/0断供与长度回退；38后台已提交d/actual5，独立985检查通过。与先前d8 bootstrap raw/noise/modelstate差0。同步同task0为87步成功/21.977s，d8为220timeout。缩短延迟恢复本次成功但多111动作步，仍不能宣称RTC质量已解决或单独定位β/延迟根因；延迟与对应掩码一起变。未自动追加β扫描/更多任务，无代码或FlashRT修改，服务已结束。详细父run/结论.md。

- 用户授权取失败任务做同步无RTC：Spatial task0/H20N5q5state80seed7init0，父run M1/20261020T054248.004291Z-rtc-handoff-comparison-30359053，worker6eea8ede87步成功/21.977s/夹爪指令切1次/0断供/请求间隔全5且全无引导；对照十任务固定d8的task0 2e52a52a为220timeout/17.241s，首sim/noise/raw差全0。说明该任务同模型配置能成功，问题集中在延迟执行+RTC引导，不单独证明β、延迟或实现根因；task0固定d5尚未测，不自动追加。原Spatial d8十任务已最终2/10（task2/7成功，其余超时）。无新代码/FlashRT/训练变更，服务已完成。详细父run/结论.md。

- 用户要求提供Spatial十任务命令自行运行，按最近固定实际d8/估计8解释，仅10任务各一次（不是d5+d8共20）。diagnose_rtc_handoff新增--task-ids 0..9与独立worker task-id，逐任务目录/保护进程/共享锁、错误即停；除task2保留已验证565–572，其余任务从bootstrap有效长度L预建L-3..L+4共至多8图，未知长度明确慢参考回退，不能承诺全批无回退或固定延迟测试为真实异步。两项配置/十worker命令与持久汇总CPU测试通过；未启动GPU，等用户执行。FlashRT不改。

- 用户ok授权固定实际d8补充，diagnose_rtc_handoff新增显式--conditions controlled_d8（默认仍原三组），父run M1/20261020T043754.313653Z-rtc-handoff-comparison-af29b922，worker dd50a031：171步success/13.529s/median575.50ms/夹爪指令切1次/0断供；20已提交后台d估计和actual均8，独立658检查通过。与前actual a3865a07的bootstrap和首guided raw逐值同，首次后台交接本组8步、actual10步，之后闭环分化；不证明所有后续raw等价。固定d8也成功，所以重点是实际9–10延迟/估计错配/余量，但未单独因果隔离。仍受控等待非真实20Hz，未调β/FlashRT/权重，无在途任务，不自动扩测。详情父run/结论.md。

- 上述三组诊断已完成：父run M1/20261020T043133.957776Z-rtc-handoff-comparison-5b05d68b；同步c6e7b6c0为112步成功/夹爪指令符号切1次，固定d5 682febd8为123步成功/切1次，实际异步a3865a07为220timeout/切21次/median582.60ms。三组首态/首noise/首raw逐值同；均0断供/图形状回退。actual已提交d8x5/9x9/10x9，相对估计8平均低估1.17步，7交接旧队列刚耗尽，碗maxlift20.13cm未放置。两RTC记录独立审计1413检查全过；不能把审计通过等同全系统正确，也不能因单次成功/失败归因β或RTC公式。受控d5非真实异步，固定实际d8尚未测。详细结论在父run/结论.md。独立服务已正常完成，没有继续扩测或改FlashRT/β。

- 2026-10-20 用户授权定位H20/task2视频动作偏差：diagnose_rtc_handoff.py仅单任务三条件（同步无RTC、受控实际d5 RTC、真实异步估计8 RTC），均H20/N5/q5/state80/seed7/init0，同原始expert/作者顺序。同步每轮bootstrap无引导，通过仅episode内临时替换infer保留正式warmup；不改生产调度/FlashRT/β、不训练、不使用子agent。逐worker768MiB保护/共享GPU锁，错误即停。控制延迟组和真实组d不同，不能用三组直接单因素归因；需结合请求/轨迹定位。结果统一独立rtc-handoff-comparison目录，不覆盖视频源cfaef1e3。

- 用户接着询问H20，按上一轮用户亲自执行方式提供命令：保持真实异步/固定估计8/N5q5/state80/task2，显式--allow-horizon-transfer现同时允许H25 gate转H20/H50（仍需现场图/eager门禁）；CLI7测试过，未替用户启动H20。上一H50 de6ca41a为220timeout/629.61ms/actual10–11步/0断供回退等待，两审计过，碗未抬起；不能说单独H导致退化，因为延迟也增加。

- 用户要自行跑H50命令：保留上轮固定引导估计8、真实异步、N5/q5/state80/task2/seed7，仅H25→50。eval_rtc_openpi新增显式--allow-horizon-transfer，只有H50 source-step author-order actual合法，允许H25 gate来源转移；不是H50独立作者body验证，setup仍逐exact长度做图/eager数值门禁，768MiB保护保留。CLI7测试过，尚未运行本次H50 GPU；不得换旧varlen或取消内存保护，等用户执行日志。

- 用户“固定8试试”在延迟估计讨论下已明确按真实异步、仅固定guidance estimate8解释并执行。新增CLI --fixed-delay-estimate（仅author-order actual合法），初始化8，history真实继续记录，裁切仍actual，不等待；默认historymax未改，去噪/FlashRT不改。aaeb19ab task2H25N5q5state80 220timeout/16.365s/median584.38ms/14.25Hz，25后台估计全8，24提交actual7x1/8x7/9x10/10x6，0断供/回退/受控等待/保护。两个审计通过。夹爪翻转13，最后碗提起20.39cm仍未放置，不能支持多估2步就是失败原因。CLI/system/算法1定向15过。详见RTC_CORRECTNESS_RECHECK第7节；未自动跑固定实际d8或估计10。

- 用户要求整套RTC核查，本轮新增audit_rtc_system（独立索引/history，不调用生产commit/DelayQueue），Spatial慢10+缓存d5+actual共12run/1967控制步/9510约束全过，state/归一化/时序/算法1counter/oldtarget/crop/动作来源未见不一致。输出actual1013a9fa/system_audit_all.json；6小记录测试能拒绝注入state/target/decode/source/crop错误，定向22过。GPU49e2bd01重放actual第8轮观测68 d12s7，同双图/state/noise/previous经完整context，eager/plain/packed及线上raw逐值同，五步VJP等全0，仅一实际请求重复核对，不是新成功任务。实际24个已提交请求21高估，平均2.04最大5步，历史max保守导致，未证实失败因果，不先改算法。前文“可能小异步bug”仍假设，当前未发现明确系统链路错误，也不证明全系统所有情况正确。详见RTC_CORRECTNESS_RECHECK.md第6节；未动FlashRT/推理/调度/功耗，未擅自追加固定d8/10任务。

- 用户固定d5 LeRobot54558bf7已160success/38.732s/1263ms/0断供，30交接d/actual/crop全5。与我方5384eecc初始图/before/noise全同，bootstrap未RTC已有raw.00374055差。后用户ok授权同KV核对：新增audit_rtc_lerobot，c620bb7f用同冻结原expert/三KV570563557、实际参考noise/previous、d=s5，逐同输入velocity差0，但LeRobot修正等于自身weightederror、相对fullVJP差.399–.402，最终raw差.241–.246，非FlashRT浮点误差。官方v0.4.4 RTC先算velocity再requires_grad输入，丢velocity-input Jacobian；CPU非线性解析测试确认。作者原函数体窄Torch桥（非JAX）在同3case五步input/velocity/clean/error/VJP/output/raw与我们源顺序参考差0。未改任一边算法，不把这个版本当fullRTC数值基准；也不证明我方真实异步调度正确/谁成功率高。新环境CPU3测试过。详见LEROBOT_RTC_COMPARISON.md最后一节，原始GPU差异run533ffccf保留。

- 用户接下来亲自跑LeRobot固定d5对照：eval_lerobot_rtc.py新增--controlled-delay-five，提前完成不早提交，执行5旧步后future.result暂停仿真，裁5步并assert consumed5；原actual模式仍不阻塞。此模式output feature配置32保留全raw32旧目标（执行仍7），避免把旧output7补零差异误认为公式错误；原生模型/RTC源码不改，noise显式保存、bootstrap及请求raw/previous/token保存。prediction_ready两项CPU测试通过，未运行此新模式GPU。给用户独立venv+guarded_run命令即可，跑后再分析；不要冒称真实异步或已通过成功率对照。

- 2026-10-20用户授权配置LeRobot并同任务试：独立venv /home/wyk/venvs/perceptbudget-lerobot约437MiB，.pth只读继承lutvla GPU库，原环境Torch2.7/TF4.53.2未变。/workspace/lerobot从原main146d847切到官方v0.4.4 8fff0fd（main需Py3.12），源码干净；新env官方TFfork dcddb970/4.53.3。新增eval_lerobot_rtc.py直接原生PI05/RTC/ActionQueue，外接LIBERO/低内存strict813项/本地SP80/state8/同动作适配，不是官方全管线CLI。5254a8ed加载及两次推理通过；a16ac148 task2H25N5q5seed7init0真实异步220timeout，median1543.28ms，官方例子ceil(wall/.05)=32–33>H25，新队列全裁空，25–219共195步断供，首次后台实际25控制步但旧队列只余20。不能当LeRobot动作质量失败/成功率/14秒速度优势；未改FlashRT和LeRobot核心源码，无OOM。详见LEROBOT_RTC_COMPARISON.md。未扩H或改官方延迟时基，等用户选择有效对照方案。环境只配推理链，未宣称全部extras/pipcheck绿色。

- 用户要求给命令自行试新版真实异步：eval_rtc_openpi仅放开显式actual-delay+author-order-reference+source-step bank组合（可packed conditioning），禁止controlled同时开/混旧varlen/fullgraph。沿用原异步调度，不改RTC/FlashRT。命令Spatial2 H25N5q5state80，exact prefix565–572预热8种，768MiB保护，遇未知state长度慢fallback可能断供或infeasible，应原样报告。CLI/scheduler/stepbank定向11tests通过，尚无此新版实际异步GPU运行，不宣称成功率/20Hz。用户将自行跑，agent只分析日志。

- 2026-10-20继续追114/115：新增camera-only audit_render_repeat.py，无模型加载。a6c0476f三个独立env回放原20动作到完全相同scene，在原sensor render内部simtime1.498固定重复20次，目标单像素RGB114→115分别4/1/1次，主要cam/geom/material/light哈希全同。1ec3af61同帧readback30次全同；GL_DITHER开/关/恢复各200次，115为6/4/1，关闭仍复现，正式GL状态恢复。已隔离到MuJoCo/NVIDIA重新绘制路径，不是保存/读取/RTC；offSamples4仅记录，未证明MSAA/具体shader根因，不声称关闭抖色修复。控制步末render与观测采样时刻不同，其大差异不可当114/115。仅新增诊断，未改FlashRT/模型/正式GL/推理。详见RTC_SOURCE_ACCELERATION.md图像端定位。

- 首次分歧已做同观测端到端回放04b20286：第20帧腕部单个像素RGB114→115（非3个像素），两张画面各自原始eager/普通图/缓存图raw与历史raw均逐值同；五步缓存图VJP等差0。相同观测有效KV及重复输出均逐值同。首次action25差.0008505，state26分歧；故本次首次预测分歧由画面输入差异触发，不是时间缓存算错。渲染端为何跨run差1级未定位，不声称具体GPU舍入因果。初轮b3008cca对全KV容量含未写尾部测repeat=false不代表有效KV不确定；v2只克隆有效范围且同步拷贝，repeat=true。只新增离线诊断script，不改推理/FlashRT/任务设置。详见RTC_SOURCE_ACCELERATION.md“首次差异原因复查”。

- 最新保真加速：可选--source-packed-conditioning需source-step bank，39个原始FP32纯时间层打包预计算，图外逐step上传live buffer（图内禁止覆盖step0）；慢fallback每步也显式select。da9c81a0三固定KV/噪声case五步raw/VJP全0，专家215–231→182–199ms。同Spatial2 5384eecc成功124步/14.589s/完整median551.49ms，上一图bf2821fc123步/15.232s/583.21ms；不是逐值同轨迹：frame20先3通道差1级，request4首次raw差.0029175但noise/previous/state同，control26才分歧，尚无相同完整观测双后端重放，勿断言纯渲染原因。24轮目标/124步来源裁切审计通过，0断供/fallback/内存保护；setup8.56秒另计。受控d5非真实20Hz/十任务。当前全套338passed9skipped178subtests，新增CLI guard另测通过。未动FlashRT/功耗/精度/attention，默认慢参考和图不缓存路径仍可选。详见RTC_SOURCE_ACCELERATION.md顶部；以下为历史上下文。

- 新单步图Spatial2 bf2821fc完成123步success/15.232s、完整median583.21ms vs本轮慢参考e6373b1a123步/43.585s/1815.74ms。25份raw/noise/previous逐值同，控制action及before场景全同；124帧仅一通道差1级，不能称images bitexact。24轮目标/123步来源裁切审计通过，0断供/重建/fallback/RAM保护。预建566/567/568花8.81秒另计；非真实异步/非十任务成功率。CLI --source-step-profile从匹配慢批次按task准备≤8长度，只shape预热；未见长度fallback慢参考。修正release日志cuda_graph_used包含step bank（已完成run旧字段false勿误判，真实requestbackend为stepgraph）。335tests已通过，新增profile测试已单测通过。详见RTC_SOURCE_ACCELERATION.md，不整套切回旧varlen路径，不自动宣称7/10赋给新候选。

- 用户授权以最新7/10慢参考为基准重新提速。原始注意力/精度/FP32源顺序不动。时间条件缓存2a52c29b三长度raw/VJP逐步差0但仅省35–60ms，暂未接入runtime。新SourceStepGraph dbfb73ca原始单步forward/VJP动态time、作者未化简gain/FP32tau原顺序，H25N5三组逐步input/velocity/clean/error/VJP/output差全0，专家1456–1458→215–232ms。初轮beta构造捕获invalid保留，v2仅常量预建修复。SourceStepBank最多8个exact长度共用scratch池，预建在setup，miss回退source-order慢参考不重建不padding，默认不开。下一步仅Spatial2一次受控q5d5与已完成慢参考123步成对比，prefix566/567/568来自该历史任务，未知长度fallback必须报告；不是已恢复全任务/真实异步加速，不改FlashRT或功耗。

- LIBERO-10 task0慢参考e8972221已完成347步success（仿真17.35s/真实125.804s/完整median1819.13ms），0断供/异常/RAM保护触发；配置H25N5q5d5state80/原attention/author_order/无图不变。任务汤罐+番茄酱放篮子，520上限。handoff_audit通过69轮实际target、347步来源/gain1/裁切，有actualdelay的340步旧动作一致；末次成功后完成未提交，actualdelay=None不算断供或伪造延迟。仅单次受控长任务成功，非真实异步/总体成功率/加速路径结果；不扩测或恢复加速。详见RTC_CORRECTNESS_RECHECK.md第5节。

- 用户要求再测LIBERO10长任务：明确按libero_10 suite task0各一次（不是十个Spatial任务），沿用源顺序慢参考H25/N5/q5/d5/state80，无varlen/图。eval_rtc_openpi新增suite选择，复用现有eval_seam_sync.get_tasks的libero_10 loader和520步上限，不套Spatial220；记录suite/taskname/maxsteps与loader源码快照。用户本轮授权agent执行；保留768MiB保护/原功耗/重复native expert释放，仅一个任务。

- 慢参考单次dde7d722已完成：Spatial0/H25N5q5d5/state80/原始attention/无图/author_order，220步timeout、80.990秒/完整median1842.92ms、0断供/无RAM保护触发，闭合70步/碗maxlift20.18cm，非真实异步。实际noise/previous保存，handoff_audit通过43个真实目标、215步pending旧动作、220步来源/gain1/裁切。9ea52871三组GPU sourcebody/Torchbridge逐步差全0；旧87dd5652 raw差.002391/.001569/1.19e-7；未证明失败因果、不称JAX/XLA/全encoder/所有输入等价。详见RTC_CORRECTNESS_RECHECK.md。默认/加速不替换，不扩测任务，未装LeRobot依赖。

- 最新用户授权由agent建立慢速正确性参考，不再扩测/调参/加速：新增rtc_author_source.py执行作者原函数体，窄Torch适配vmap/scan/vjp（非JAX/XLA验证、共享同原始expert/KV），audit_rtc_author_source。旧87dd5652原始expert H25三case有一例raw .00156857（末步VJP差.00184537）；显式author_order候选按源tensor标量/clean乘法顺序/未化简gain/累加tau，9ea52871三case每步input/velocity/clean/error/VJP/output/mask/raw差全0，冻结原始权重，无varlen/图。默认旧路径不自动换；--author-order-reference仅受控d5禁加速/actual，必须新GPU gate/作者SHA一致，完整保存noise/previous_raw。历史29152932离线oldtail/裁切/来源220步通过但目标输入未存，不能当完整端到端证明。下一步仅一个Spatial0慢参考任务并审计实际输入，不称官方完整复现/失败根因已定。

- 用户要求尝试调用LeRobot。本地clean checkout146d8474932c8b566717eb2738ba459337ab71c4，完整RTCProcessor导入缺draccus，未安装/改环境/未启动GPU。新增audit_lerobot_rtc.py仅AST原class body（配置/enum/processor）CPU执行，非完整LeRobot策略或任务。af77fcf4 toy v=2x/time=.6，源码先计算v再requires_grad，捕获velocity输入requires_grad=false，correction=.2而完整VJP应-.04；不改源码，不据此断言用户异地版本相同或当前RTC已正确。完整policy验证仍待依赖与兼容性检查，不能把source-class小测当LeRobot任务复现。

- 用户授权修正RTC延迟历史：Algorithm1的A_init bootstrap不入Q，Q仍初始化[d_init]；DelayQueue.observe显式observation_step=0跳过，不把正常后台0步延迟一并过滤。harvest记录bootstrap排除原因和delay_queue_observed=false。新增初始保守值保留到正确10个样本后淘汰的回归测试。用户自行重跑H25/N5/q5/state80真实异步+释放重复native expert，agent不启动GPU，不改变attention/精度/H/N或宣称数值等价。

- 用户要求H25：入口--horizon增加25，非默认H仍只允许显式actual-delay实验，reference_horizon_transfer记录真实H，不冒充H20独立参考覆盖25。保持释放重复native expert、N5/q5/state80/Spatial0/原功耗/768MiB保护，用户启动GPU，agent仅给命令和分析。默认H20不改。

- 用户将自行重试H50内存清理：新增rtc_memory.py和显式--release-unused-native-expert，仅预热完成的fixed BF16管线，sync后删除_ckpt_bf16 decoder_*重复权重/对应指针，保留encoder/vision/context graph/KV/冻结原专家；进程不可逆context-only，禁native infer/infer_batch/rebuild及full/decoder图replay。保留native图handle而不手动destroy避免共享所有权风险，不称释放全部图内存。每阶段memory.jsonl与native_expert_release.json记录预计释放权重bytes与实测内存；episode reset前gc/empty_cache。CPU guard/保留测试通过，未在GPU验证、不代用户启动，768MiB保护不变。代码只在主项目，不改FlashRT/精度/权重计算/H/N。

- 用户授权H50实验：eval_rtc_openpi --horizon 50仅显式actual-delay，默认H20不变；native warm和余量检查均用config H。沿用H20参考门禁只确认同checkpoint/源码/N，H50迁移单独记录reference_horizon_transfer，不冒称H50独立数值验收；live graph仍核对同后端。保持N5/q5/state80/原功耗/真实异步/768MiB保护，给用户命令，不代启动GPU。

- 2026-10-19 用户将亲自运行真实异步候选，要求给命令/后续分析日志，不由agent启动GPU。eval_rtc_openpi新增显式--actual-delay，仅与--experimental-varlen --cuda-graph组合，禁止controlled-delay-five/timing-only/verify组合；默认不改，保留原版独立门禁但不冒称实验attention通过。实际模式不设controlled_delay_steps，paper delay queue按实际步数，gain1/原仿真/768MiB保护；H20N5/state80/Spatial0。是否足够H余量仍由预检/运行记录判断，不承诺无断供或稳定20Hz。尚未实际运行本模式。

- 2026-10-19 用户批准KV顺序对照及任务小测。context-first exact只换内核raw差.0147674(15ff257f)，action-first exact .00697818(45cc834d)同cap592，仍未过1e-4，不放宽门禁。显式controlled-delay-five/experimental-varlen仅质量入口，禁实际异步；图门禁标同后端。父d3bf8e06仅当前Spatial0/H20N5q5d5state80，不是历史失败严格复现：original dc9a71e0成功144步51.335s/1831.95ms；varlen7096c07a成功146步17.542s/570.25ms。初始before完整场景同、首raw差.00171929，首图/KV未独立逐位比较。二者570–573有效长度、0重捕获/断供、真实delay均5，执行来源gain1全审计。候选保留显式实验、默认不改、不批测，不把受控wait叫真实异步；详细RTC_ORIGINAL_REFERENCE.md末节和父analysis.md。首编排仅标签超32字符启动前退出，v2修正，无episode丢失。未改FlashRT/功耗/H/N/权重，无子agent/训练。

- 2026-10-19 用户授权仅移植旧变长attention到原始expert。openpi_varlen.py仅替换隔离source模块attention（不改FlashRT/安装依赖），固定cap592动态cum_k、原始其他计算不动，probe显式--varlen-attention。896d1ea7同图570/563/557/570，samebackend graph/eager与padding毒化差全0，专家192.4–192.8ms；独立原始eager raw差.006978，五步velocity/VJP也未过1e-4，failed_gate，不接默认/不跑任务/不宣称成功率。第一轮1a288ac6容量字段错误valid-only实际592，后轮修正。320pass/9skip+178subtests。详见RTC_ORIGINAL_REFERENCE.md末节。下一步须解释attention数值差异/质量影响，不假称变长实现已经保真通过。

- 2026-10-19 用户要求修state换长度卡顿。当前取代e85cb88自动重捕获：仅保留一张exact图，其他长度回退原版eager（共享权重、独立KV），恢复原长度复用图，重复prepare明确拒绝。完整44a69f7a合成state切换573/557/573/570，raw/VJPnorm差全0，0重建，完整1856/1823/1818/563ms；原长稳态median563.51ms，初始化4.97秒另计。只是取消6秒建图停顿，不是所有长度保持564ms/真实异步，CLI仍timing-only。多图完整9c21b280触发765.8MiB保护未开始任务，错误JIT和全模块图不采用；仅FFN图虽exact但约1.31s未集成，废案源码均在任务产出source，主干不留。未改FlashRT/模型配置/功耗、未训练/用子agent。详见RTC_ORIGINAL_REFERENCE.md最后小节。

- 2026-10-19 用户要求先state再复查旧加速失败路径。exact-length单图释放/重捕获适配新增，563/569/570三次原版核对raw/VJPnorm差全0（44897537），但准备6.32–6.34秒，不能宣称连续异步可用，CLI仍仅timing-only。旧路径独立复查1038e871共享KVexact但fused/unfused首步速度/VJP max .016214/.014528，reference-projections仍.012237/.013376，未达1e-4，不融合旧内核。没有实际任务成败复测，不称已复现原版任务成功；详见RTC_ORIGINAL_REFERENCE.md。FlashRT/H/N/精度/state容量/频率不改，无子agent/训练。

- 2026-10-19 用户授权原始专家保真加速。新增openpi_graph.py/probe_openpi_graph.py，原始计算+作者Euler不改，仅固定FP32 timestep与隔离suffix局部固定mask常量提前创建（全局torch不改）。exact prefix图保留context强引用、替换地址拒绝；可原地更新同长度KV。f275749a专家普通1449.84→图208.83ms，5步速度/VJP/raw独立差全0、动态noise/target/d/s与KV内容差0且输出改变。完整79c21c1f固定观测计时573.29ms（三次568.96–578.25）vs此前1837.19，live gate raw/norm差0、建图4.86s另计。CLI --cuda-graph强制仅--timing-only，不运行变长state episode、不报成功率/Hz。固定pad592方案55e51f7a相对有效长度独立raw差.008165未过，不采用；首次578cc0ee因mask构造捕获invalid保留。运行均有768MiB保护未触发，未改FlashRT/默认/H/N/精度/功耗。315pass/9skip+178subtests。详见RTC_ORIGINAL_REFERENCE.md追加；不要把专家209ms当完整请求或把573ms叫真异步成功。下步需保真处理实际prefix长度与继续压延迟。

- 2026-10-19 用户批准FlashRT编码+OpenPI原始专家正确性参考。3b224fc新增openpi_reference.py/rtc_openpi.py/eval_rtc_openpi.py，隔离原始Gemma模块，AST原方法体不重写；原checkpoint suffix混合精度冻结，只共享prefix KV，bootstrap也原专家。作者Euler顺序canonical gate d9835f75对独立7c7d1e10五步速度/VJP/最终raw全0，源码SHA门禁；首gate a85cb455拆分Euler参考raw差.005876未过保留，不放宽门槛。实际task0预检571c2e57完整eager 1837.19ms、20Hz初始delay37>H20-s5=15，状态infeasible_before_episode，无正式episode/成功率/断供，正常结束未触发768MiB保护。参考用heuristic-first，不与393ms直接比较性能。仿真优化关闭，未改FlashRT/依赖/默认/H/权重，未训练；312pass/9skip+178subtests。详情RTC_ORIGINAL_REFERENCE.md。后续需保真执行加速才能验证真实异步，不能假称已完成。

- 2026-10-19 用户要求继续定位：逐层首差在动作输入投影（项目BF16 matmul先舍入+加bias再舍入 vs原OpenPI FP32投影转BF16）；相对.2839%/max.03125，独立参考只换同顺序后与项目该层逐位同。第一层Ada差1.406%；逐层trace41ec3e4a。固定首步相同残差单项消融：输出dt折叠clean-VJP差28.79%、输入BF16 16.51%、timeMLP BF16 31.00%，不可相加/不能当成功率因果。诊断方法enable_reference_projections默认关闭，仅原4个FP32投影权重，输出非预折叠；cf5445c7未融合首步梯度差52.57→43.96%，尚未对齐，不接正式CLI/不批测/不称已修复。仍需time style/Ada/激活残差/attention精度逐项核对。309pass/9skip+178subtests；无FlashRT修改。详见RTC_CORRECTNESS_AUDIT.md首差消融追加。

- 2026-10-19 用户强调核对算法并实现正确pi05异步。新增audit_openpi_expert.py隔离加载OpenPI原始Gemma源码+原始checkpoint专家，FP32动作/time投影、norm调制FP32、其余BF16；不安装/改transformers。录制prefix KV先逆FlashRT RoPE交错排列。独立参考run4bfa20b7最终raw相对融合已录1.630%/max.019086；同每步输入/残差且18层KV逐位门禁通过的cb3b3e95，速度场差.43–.69%，clean-VJP差13–52%，关闭融合仍近似同差。恢复底层速度VJP差.45–.73%，相减放大不能直接判反传公式错误，仍未等价通过/未定位首层差异或失败因果。新增Algorithm1时间线独立测试通过（非完整竞态/20Hz证明），308pass/9skip+178subtests。详见RTC_CORRECTNESS_AUDIT.md；不批测、不启用仿真优化、不称393ms版本正确性验收。作者JAX全模型未跑；已跑OpenPI专家但encoder未独立验证。

- 2026-10-19 用户批准先正确性对照不批测成功率。新增audit_env_replay.py，同原task0的176步动作执行原/优化/原：da5a45be两次原state逐位同、176帧同；优化state最大差4.5633/control2.7791、仅59帧同，零基59开始>1e-6，三次均成功。确定forward复用不满足运动轨迹等价，仍默认关闭，不能用其2/10评价RTC；无优化4/10仍需查。作者仓库只读下载任务产出/rtc-author-reference，9296f31d，JAX无安装；audit_rtc_author_mask执行作者原函数AST+NumPy1557例差4.44e-16，仅Eq5。官方π0.5专家/VJP独立验证尚未完成，不称已复现、不自动批测。详见RTC_CORRECTNESS_AUDIT.md。

- 2026-10-19 仿真优化版RTC十项已完成：父run=M1/20261019T065322.242512Z-rtc-async-ten-envopt-2e3da771；有效10/10，2/10成功(task5/7)，其余220步超时，全0断供/无异常退出。task5 136步8.673s、task7 133步8.581s，成功平均8.627s。控制16.07–18.31Hz；257次请求合并推理中位425.20ms。skip50725=2029步×25，来源2029步逐位核查、gain1、实际裁切索引与actualdelay一致；历史队列257次核查通过；实际延迟248项中同124/高估117/低估7，不能把生成约束偏差称执行跳错。旧无优化十任务4/10；尚无移动/接触的仿真优化等价、跨启动选核控制，不把下降单归Hz，不替换默认，不自动扩测。结果summary.md/audit.md；运行中记载已由本条完成状态取代。

- 2026-10-19 用户“再测十次”按刚才17.7Hz仿真优化版理解并已说明：388529d扩展eval_rtc_async_screen.py --reuse-control-forward，Spatial0–9全新各一次，不复用旧数据；H20/N5/state/80/s_min5/队列max10不变，每项skip>0与模式隔离强制审计，记录延迟估计偏差数。服务perceptbudget-rtc-envopt-ten-20261019；日志/tmp/perceptbudget-rtc-envopt-ten-20261019.log；父run=M1/20261019T065322.242512Z-rtc-async-ten-envopt-2e3da771。guarded_run保护不变，异常停批次，可resume。运行中不改推理/控制代码，不预报最终成功率，不冒充稳定20Hz，不自动替换默认。

- 2026-10-19 用户让测优化后实际Hz，单Spatial0 RTC真实异步保持H20/N5/state/80/s_min5/融合快版。03f1a43首次在reset前安装forward复用，reset重建sim导致skip0，c79832f6成功184步13.239s/14.23Hz，不算优化证据。f04cb11改为run_episode重置后安装且强制skip>0；665404de实测220步超时12.712s、17.73Hz、env.step50.84ms/tick56.32ms、skip5500、0断供、推理中位424.84ms，预检/来源审计通过。旧十任务task0为成功176步12.735s/14.15Hz，不能用失败wall秒说完成提速。优化默认关闭、仅进程内不改依赖/FlashRT。307pass/9skip+178subtests。尚无移动接触逐步等价/因果定位，不自动扩测或替换默认。完整analysis.md在M1/20261019T064817.114511Z-rtc-paper-schedule-q5-state-665404de。

- 2026-10-19 用户问为何env.step慢及能否改进。新增仅诊断scripts/probe_env_step.py，无模型，Spatial0固定动作各30步，保持2ms物理步长/25子步/两路256图像。M1/20261019T064241.131452Z-env-step-profile-7cfd838b：默认55.139ms，BLAS8→1仅54.976ms；主要forward21.50/physics11.66/controller9.39/dispatch5.43/cameras4.72ms。发现外层forward和controller.update每子步各一次；进程内试跳过控制器内forward后44.637ms，省10.50ms，逐步state最大6.63e-15/ctrl2.78e-17/双路图像全一致。仅固定动作短探针，未改正式runner/依赖/FlashRT，不能保证移动接触等价或并行20Hz；还需外部约6ms及资源争用。原始结果和analysis.md已落盘。首探针25444746缺threadpoolctl建环境前invalid保留，后不安装依赖而用加载OpenBLAS接口；中间bf499175仅默认/单线程。没有自动重测成功率或提升正式默认。

- 2026-10-19 Spatial纯异步十任务批次已全部完成（5e0a947编排，推理bf13753）：父run=M1/20261019T061721.555052Z-rtc-async-spatial-ten-10cc127a。10项全新测有效10/10，成功4/10(task0/2/6/7)，其余220步超时，全部0断供/无异常退出。任务步数176/220/134/220/220/220/140/136/220/220；成功平均回合真实10.599s；非bootstrap292次请求合并中位417.62ms。控制13.19–14.90Hz，非稳定20Hz。历史队列max292次核查通过，1855步pending时执行、gain全1、来源worker审计通过。H20/N5/state/80/s_min5/队列10保持，不复用三项、无模型/FlashRT修改。结果summary.md及audit.md。当前不能替代旧M1正式基线，未自动扩测/调参。

- 2026-10-19 用户要求Spatial全部10任务纯异步成功率。5e0a947新增只编排的eval_rtc_async_screen.py，10项全部新测不复用前三项；H20/N5/BF16/state/80/融合快后端、算法1队列max10、s_min5、原速无固定冻结，seed7/init0。唯一服务perceptbudget-rtc-async-ten-20261019，日志/tmp/perceptbudget-rtc-async-ten-20261019.log；父run=M1/20261019T061721.555052Z-rtc-async-spatial-ten-10cc127a。每任务guarded_run+GPU锁，异常停批次，可--resume-from只续缺项。运行中禁止改推理代码；实际控制Hz单独记录，不承诺20Hz，未完成不能报最终成功率。原始三任务仅历史对照。

- 2026-10-19 用户要求试论文RTC调度，a2b00bf新增--paper-schedule/rtc_schedule.DelayQueue：s_min5、单线程推理空闲即刷新、最近10个实际延迟步数max，初始d由preflight/.05向上取整。删除该模式额外s>=估计d等待，bootstrap d>s允许但必须d<=H-s（公式不变，不裁剪；图3稳态约束初始化边界明确记录）。同Spatial0/1/2新结果成功182步13.163s/超时220步15.470s/成功141步10.109s，2/3，全0断供；任务内推理中位423.59/418.42/423.91ms，控制14.16/14.51/14.40Hz，非稳定20Hz。history逐项核查28/34/21次通过、gain1、来源审计通过；初态hash同旧但bootstrap raw差.00390625/.00390625/.0078125，不能把改善全归调度。记录M1/20261019T060802.697243Z-rtc-paper-schedule-q5-state-ff4383a5/three_task_summary.md，另run bf82994a/0e458cc0。306pass/9skip+178subtests。未训练/未改FlashRT/未替换M1或自动扩测。

- 2026-10-19 用户授权真实延迟异步三个Spatial任务0/1/2各一次。03eb4a9新增--actual-delay，复用原入口不设controlled_delay_steps；q5为最小间隔，保留d<=s约束动态推迟，实际间隔5–7步，原速gain1。H20/N5/state/80/flash-varlen融合快后端。结果0/1超时220步15.596/16.149s，2成功140步9.751s，全部0断供，推理中位421.25/425.29/425.69ms，实际控制14.39/13.89/14.83Hz，不能称稳定20Hz/正式替代旧M1。env.step平均63.73/65.94/61.59ms，控制准备+尾部约6ms；未细分内部物理/渲染或证明GPU争用。完整记录在M1/20261019T055901.803138Z-rtc-actual-delay-q5-state-54bea0c0/three_task_summary.md，另外run后缀09243f5a/cce148fa。无额外扩测或FlashRT改动。不能把超时耗时叫成功完成时间。

- 2026-10-19 用户同意先稳定选核/80容量再优化。新增项目侧可选native_gemm_policy=heuristic-first，仅代理autotune候选数1，不改FlashRT，默认autotune保留。80a/80b固定fixture独立进程所有追踪阶段逐位一致、padding检查通过；但首候选完整RTC477.01ms/普通392.48ms，不能作为加速默认。同80自动选核固定输入完整RTC392.87ms(390.34–394.47)/普通304.33ms，context259.97/专家122.79。不是任务/真实20Hz结果、不是自动选核跨进程一致性证明。要保留最快算法并固定需原生导入导出接口，尚未实现。CPU304pass/9skip另178subtests。未训练/未扩测任务/未继续改编码器。详见RTC_CAPACITY_DIAGNOSIS.md后续记录。

- 2026-10-19 用户确认文字/state容量选80：config.json新增state_prompt_capacity=80，作为新实验默认，图像token另计、完整分词不变、超过80报错不截断，CLI仍可显式覆盖用于诊断。历史run和64容量数据保持原样，不把393/425ms归给80。仅做配置/边界测试，不自动重跑GPU任务；本次不改模型/H/N/state开关/FlashRT或动作衔接。

- 2026-10-19 用户要求检查capacity200→64为何改变轨迹：仅新增diagnose_state_capacity.py及比较测试，不改生产/FlashRT/权重。固定同图/state/token/noise/target，200/64/200隔离对照。首轮有效encoder/RTC全逐位一致；追加bootstrap轮64自动选核encoder第0层FFN down日志best1→5，首次差异在该GEMM输出、输入逐位同，200跨轮与重复同。三worker位置/有效长度一致，encoder/RTC padding污染检查均不影响有效输出。native bootstrap首层注意力输入相同、输出有差异；固定seed7 native首块cap200/64 max .0078125，各自逐位匹配历史104步/187步任务首块。不能把全部变动只归容量舍入/只归splitKV，Torch确定性不锁原生autotune。详见RTC_CAPACITY_DIAGNOSIS.md；原始run M1/20261019T053108.377723Z-capacity-numeric-diagnosis-aeab67f7及M1/20261019T053555.980538Z-capacity-numeric-diagnosis-c81702d8。首探针OpenPI导入失败invalid保留。不自动修选核或扩测任务。

- 2026-10-19 用户要求尽量压到400ms。项目侧compact_state可选固定prompt容量200→64，原200分词/完整图像/state保留，实际prompt>64明确报错不截断，不改FlashRT/default。新CLI --state-prompt-capacity 64和可选--profile-rtc-stages。task0 compact64成功187步16.391s，任务内median424.91/p95432.12ms，preflight403.18ms，参考差0；较旧104步9.62s轨迹更长，不说任务更快。另固定输入同配置5次387–395ms/median392.51ms，普通同compact64为316.81ms；只能说固定输入低于400，任务中/跨运行未稳定达标。当前瓶颈图像+context约261ms/专家124ms/其余约6ms。原生梯度GEMM+432MiB cache没有加速，已从主代码撤去，原始代码/结果归档M1/20261019T051905.731157Z-rtc-native-ffn-gate-77b30896/source。历史十任务仅FFN组53–61文字token，本task58–61，64容量不是通用保证。308tests/9skip。详见RTC_NATIVE_PROBE.md，默认不变，不自动扩测或换计时口径。

- 2026-10-19 用户要求继续压缩RTC：新增可选 fused_attention_ops，项目侧融合QKV/RoPE、门控残差前向/VJP，注意力/最终Ada用原生前向+融合反传。FlashRT、H20/N5/q5/d5/state/权重不变，默认关闭；CLI必须native-fused-ffn/preflight。设备专项5通过、305默认tests/9skip。固定输入专家160.87→126.39ms，完整含context波动较大；同后端参考raw/VJP差0，但旧仅FFN后端raw relL2 .3504%/max .0039000，非严格等价。同task0新104步成功9.620s、20次median448.33/p95454.99ms、0断供/0建图，104步来源审计通过，初态/首帧匹配旧148步14.434s。时间改善不能全部归因计算提速，不将此前8/10成功率赋给新后端。结果M1/20261019T051135.428872Z-rtc-controlled-d5-q5-state-1d2739a9，固定输入gate M1/20261019T051007.266965Z-rtc-native-ffn-gate-fddb9d64；记录RTC_NATIVE_PROBE.md。仍受控非实际20Hz，不自动扩测/换默认。

- 用户授权Spatial十任务成对比较新未融合flash-varlen与native-fused FFN，固定N5/H20/q5/d5/state/seed7/init0，不与旧flash后端混称。scripts/compare_rtc_ffn.py复用配置审计通过的task0/1四条记录，补其余16条，按任务交错模式顺序；每子进程guarded_run/GPU锁，独立run，不改在途推理代码。父run=M1/20261018T160739.023376Z-rtc-ffn-paired-spatial-0dfc5544，服务perceptbudget-rtc-ffn-ten-paired-20261019，日志/tmp/perceptbudget-rtc-ffn-ten-paired-20261019.log。异常停批次，--resume-from父run只续缺项，旧无效尝试保留。初态sim/首帧hash配对核对，超时耗时不是完成时间。启动时已4/20记录，后续状态以summary.json为准；未完成不得报最终成功率。

- 2026-10-18 用户授权融合后端完成一次相同Spatial task0：N5/H20/q5/d5/state/seed7/init0，--native-fused-ffn显式开启、强制preflight与flash-varlen/single rtc-only；默认不变。结果M1/20261018T155540.802683Z-rtc-controlled-d5-q5-state-6ee45566，148步成功14.434秒，29次稳态median473.42/p95482.77/max488.38ms，0断供/0图重建，148步来源审计通过。固定输入完整447.37ms/专家159.08ms，参考raw/VJP差0；建图4.63s+预检9.46s单列。未融合旧task0为195步20.04s，不能把28%任务时间变化都说成计算提速或泛化成功率提高。仍受控d5非真实20Hz，不自动扩测。详见RTC_NATIVE_PROBE.md。

- 2026-10-18 原生FFN试验后用户授权继续融合反传。新增项目侧 rtc_fused_backward.cu/.py，仅Ada/GeGLU输入梯度，构建产物写结果root/build，不改FlashRT。可微expert显式enable_native_fused_ffn，默认关闭。N5固定输入完整RTC专家181.46→159.03ms、context+专家466.46→444.40ms；同后端独立参考误差0，但相对旧后端raw max0.0049215/relL2 0.3741%、VJPnorm max0.0034025，未过1e-4等价gate，不替换默认/不测任务/不称成功率不变。CUDA专项3通过，默认303tests/7skip。详细记录RTC_NATIVE_PROBE.md，后续若继续必须明确数值后端变化。

- 2026-10-18 用户授权原生专家前向加输入反传的单层试验。已有 GitHub baseline cbed4c8/tag rtc-before-native-backward-20261018。新增 rtc_native_probe.py 和 probe_rtc_native_ffn.py，仅探针，FlashRT/默认RTC未改。真实权重/style+合成隐藏输入，三组FFN输出与VJP含/不含残差核对通过探索门槛；原生前向+Torch解析反传约0.934–0.940ms，旧约0.963–1.049ms（末组波动），初轮约2–3%收益。结果M1/20261018T153803.611197Z-rtc-native-ffn-gate-3a79f6b6。不是整个专家或完整RTC替换，不能声称全模型提速。两次中间跨流捕获invalid保留，最终同流创建叶子/梯度种子解决。300tests通过/4skip。详见RTC_NATIVE_PROBE.md，暂不扩大任务。

- 当前用户授权压缩带state RTC耗时：新增可选flash-varlen项目侧后端，用Torch已有FlashAttention变长前向/反向及GPU累计长度，让单一固定容量CUDA图排除padding、适配state有效长度，不改变RTC公式或checkpoint，不改FlashRT源码。先GPU小张量前向/VJP/同图变长重放验证，再真实模型独立参考gate及计时，最后仅同一个Spatial task0验证。N5/q5/d5/state/确定性保持，不冒充真实20Hz异步。preflight-rtc将一次性核查计时单列，任务计时不混入核查；报告仍需说明受控等待及后端数值可能不同。

- 磁盘帧补测task4仍在164步后触发内存保护，不能归因于保存期或宣布内存已修复。获用户“继续”授权后，同state测试可选--eager-rtc：不创建RTC CUDA图，沿用同一确定性Flash SDPA及已核对RTC公式，每步仅一次前向+VJP。续测4–9，前4项保留原图版本有效结果；resume摘要显式标注prior/current solver backend，不能将混合执行结果当作全图快版性能或同质完整性能基准。先验证task4数值和内存，失败立即停，不关闭768MiB保护。

- 最新state续测task0–3均有效成功；task4在第219步后、压缩前触发739.5MiB保护，说明只在任务后释放图不足。state单回合改用磁盘映射保存原始帧，定期flush/回收映射页，npz保留相同像素/格式且不再将整段帧复制叠入RAM；不改推理或控制。新run通过--resume-from复用前4个有效记录、续测4–9，旧task4无效尝试保留，768MiB保护不变。

- 最新state批次在task1完整220步后、结果落盘前触发内存保护。修正只在executor结束后、CPU帧数组压缩之前释放已无用途的RTC CUDA图，记录释放前后MemAvailable/torch allocated/reserved，保留768MiB保护。补测用--resume-from旧批次，仅复用有效task0成功，旧task1无效尝试保留；新run明确记录source_task_dir与版本修订，续测task1–9，不改变推理/动作参数。

- 用户要求重试state RTC：项目侧Flash SDPA适配为有效prefix长度改变时先释放旧图，再按真实长度重建同后端图，不保留多长度图、不忽略padding、不换后端。每种新长度首次独立参考核查，记录valid_prefix_length/context_graph_rebuilt/context_prepare_seconds；重建和参考核查可能显著增加时延，报告不能只报复用图稳态耗时。重测仍N5/q5/d5/H20/state开启/Spatial十任务，invalid立即停批次，旧无效run保留。

- 当前授权：按最近q5的相同Spatial十任务配置，仅开启state，N5/H20/q5/d5、确定性Flash SDPA、seed7/init0/220步不变。入口加--state，沿用归一化8维state进入FlashRT prompt（不是新增连续state投影）。先以task0检查兼容性，数值或有效prefix长度不符则批次明确停止，不静默换后端，不重复启动无效十任务。q5无state对照7/10、稳态median457.18ms/p95469.08ms，0断供。

- 当前授权由q6改为q5：不测q6；在相同Spatial十任务各一次上测试N5/q5/d5，其余沿用q8（H20、state关闭、确定性Flash SDPA、seed7/init0、220步）。q8已完成7/10成功、0断供，191次稳态请求median458.43ms/p95466.27ms，十任务数值/来源审计通过。q5独立run不覆盖q8；仍受控延迟，不是真20Hz部署。

- 当前用户授权：LIBERO Spatial十个不同任务各一次，RTC N5/q8/H20、固定延迟d5（必要时暂停仿真），state关闭、确定性Flash SDPA、seed7/init0，Spatial标准上限220步。入口eval_rtc_controlled.py --spatial-screen --rtc-only --suite libero_spatial --q 8 --denoising-steps 5 --attention-backend flash --deterministic-attention。逐任务隔离进程防内存累积，统一父run目录，保存有效/无效与数值/动作来源审计。不称真实20Hz异步，不训练，不自动增加任务次数。

- N5单任务已完成：run M1/20261018T055233.704749Z-rtc-controlled-d5-q10-86c85680，433步成功、断供0、来源审计及同后端参考核查通过（raw/VJP差0，norm长度5）。42次稳态完整请求median461.51ms/p95464.84/max469.54，较N10同口径632.16ms少27%。但N10成功353步，N5多用80控制步；不只凭推理快宣称动作效率更好。仍受控d5，不是真20Hz部署，不自动扩测。报告M1/RTC_五步去噪同任务测试_20261018.md。

- 当前授权：同一LIBERO-10 task0仅将RTC去噪N10改N5，入口eval_rtc_controlled.py --rtc-only --attention-backend flash --deterministic-attention --denoising-steps 5；H20/q10/d5/state关闭/seed7/init0不变，单次受控任务及实际请求计时，不称真实20Hz异步。全局N仍10，不改FlashRT或模型权重，不训练，不使用子agent。

- 2026-10-18 RTC优化路径修正复测完成：FlashRT c678efe修正图内速度重建顺序，bea4db9补验证说明。math与确定性Flash SDPA各自同输入图/参考raw最大差0；后者5次重放差0，专家346.92ms、context+图605.29ms，任务内完整请求稳态632.16ms。`eval_rtc_controlled.py --rtc-only --attention-backend flash --deterministic-attention` 同task0/d5/q10在353步成功，断供0、来源审计通过；math本轮仍520步超时（bootstrap相对历史成功raw差.0078125）。默认非确定性Flash SDPA未过严格核对，不省略确定性选项宣称已验证。不称实际20Hz部署或全局成功率恢复。报告 M1/RTC_优化路径修正与复测_20261018.md；290项全套测试通过（3默认skip），15项RTC专项设备侧全部通过。不再自动扩测。

- 2026-10-18 用户授权修正优化RTC路径：保留参考版速度重建/反传顺序、统一FP32设备软掩码与系数舍入；先同输入数值gate，再测Flash SDPA耗时与同一受控任务。只改可选RTC，不改普通推理/SEAM/模型权重/全局默认后端，不训练，不使用子agent。Flash SDPA固定有效prompt长度（state关闭），不静默忽略padding。

- 2026-10-18 两版RTC差异已在同输入隔离：旧eager与旧CUDA图逐位相同；只将estimate由 `x+(t*N)*delta` 换为 `x-t*(-N*delta)` 即重现参考版首个guided chunk（相对旧max .00731444，相对参考5.96e-8）。第3次去噪首次VJP分叉，源于FP32缩放顺序令一个梯度跨过BF16舍入中点；不是CUDA图自身或VJP接口错误。诊断run `M1/20261018T052343.180443Z-rtc-difference-isolation-e1ce4d41`，报告 `M1/RTC_两版本差异定位_20261018.md`。本轮仅诊断，不改生产/FlashRT路径，不由单任务一成一败宣称数学逻辑错误或普遍质量优劣。

- 2026-10-18 RTC排查结果：H20/N10/q10/d5/state关闭，慢速 `scripts/eval_rtc_controlled.py --rtc-only --paper-reference` task0在354步成功，原图RTC520步超时、无引导485步成功。初态/首帧/首chunk完全一致，首个guided raw最大差.00731444。FP32解析测试通过，但BF16图与独立参考未过1e-4严格门槛；不能把数值差直接宣称已定位逻辑错误。参考路径不使用完整专家CUDA图，约4.914s/请求，非实时部署；不切全局默认。报告见任务产出 M1/RTC_逐步论文参考核查_20261018.md。两次核查内存中断已标记，不计失败率。

- 当前另获授权：SEAM结束后做 RTC 单任务受控延迟对照，不追求实时速度。入口 scripts/eval_rtc_controlled.py，LIBERO-10 task0，各一次无引导/真实VJP引导，同一math专家、H20/N10/q10/d5/state关闭。仿真执行5步后等待结果、禁止提前提交；延迟不再取墙钟估计，不称实时异步成功率。失败后需核查论文2506.07339v2，单次失败本身不是实现错误的证据。

- 2026-10-18 用户要求不再使用子agent，后续由主agent直接实施与核查。
- 当前授权：仅给正式异步 SEAM 4/10 配置开启 state，重测相同 LIBERO-10 十任务各一次。H20/N10/K10/M10/λ0.1、seed7/init0/520步与衔接不变；使用 eval_seam_async.py --state，不改 FlashRT 或全局默认配置，不训练。

## 当前开发目录（2026-10-17，优先于下方历史）

- 当前用户授权：测 H20/N10/K10/M上限10/λ0.1、无额外 state 的 SEAM 实际异步成功率，LIBERO-10 十任务各一次。入口 scripts/eval_seam_async.py，单任务进程依次运行并持有 GPU 锁、内存保护；推理时原速执行旧队列，提交按实际延迟裁剪。记录实际 M、断供、并发和来源，不将同步 9/10 当异步结果，不训练。
- 上述正式异步测试已完成：run `M1/20261017T154039.894474Z-seam-async-h20-k10-8b6f2035`，4/10 成功、6 超时、0 断供；十任务均通过动作来源/真实并发审计。生成时完整 M10 引导，实际提交后剩 2–4 步引导重叠。此前移动掩码的 3 任务变体已停止保留，不混入正式结果。不因超时自动调参扩测，不把 SEAM 切成默认异步衔接。报告见任务产出 `M1/SEAM_H20_异步十任务结果_20261017.md`。

- 用户要求减少工作区：代码修改集中在 `/home/wyk/workspace/PerceptBudget` 及 config.json 对应的 `/home/wyk/workspace/FlashRT-upstream`，当前开发分支均为 `project/perceptbudget`。
- SEAM 同步运行器已整合到主项目，默认读取 config.json 的 FlashRT 路径。旧 `PerceptBudget-seam`、`FlashRT-seam-native` 工作区归档，不作为开发或运行入口。不要为常规实验另建并行目录。
- FlashRT 每次增删改文件，更新其根目录 `PERCEPTBUDGET_CHANGES.md`，写明文件、用途、启用方式和验证状态。官方参考基点为 e4fe123；RTC 与 SEAM 均为项目本地扩展。
- 已有 RTC 工作先存入 Git 后保留；合并时保留主项目 RTC expert/probe 实现，当前 SEAM 使用独立同步入口，早期异步 SEAM 探针只留历史。
- 结果统一写到 config.output_root。此次整合不训练、不覆盖历史结果、不更改全项目 H/N/state/异步衔接默认值。

## 当前授权：参考 LeRobot 优化本项目 RTC（2026-10-17）

- LeRobot 仅作只读参考，不在 LeRobot 侧实现或安装运行环境。RTC 实现和优化留在本项目的可选 FlashRT 接入路径。
- 用户同意继续处理明显偏慢的 RTC。新增 PyTorch Flash SDPA 可选后端；`math` 仍是默认。先用同一固定观测比较 math/flash 的完整 N10 CUDA Graph 输出、VJP、耗时与显存，不跑任务，不切换默认。
- 合成 BF16/GQA 探针及真实模型固定场景对照已完成：run `M1/20261017T050552.262146Z-rtc-sdpa-backend-ebde4f12`，math/flash 新旧动作一致输入下的RTC原始输出相对RMSE 0.240%；上下文刷新+RTC中位数870.78/599.40ms，Flash快31.2%，但仍慢于原版推理371.88ms。无任务成功率结论；默认仍为math。详情见`../任务指导与相关参考/PerceptBudget_RTC_FlashSDPA后端对照_20261017.md`。LeRobot RTC本身没有可直接复用的反向加速后端。

## 当前授权：完整10步RTC图测试（2026-10-17）

- 用户确认继续测试，先完整N10生成与端到端context耗时/数值gate，不跑任务。实验分支不变。新增CapturedRtcChunk同一graph实际包含k0..9各步VJP，不重放step0冒充N10；旧单步入口/默认路径不变。
- 固定task0观测、合成remaining raw目标d7/s8用于数值对照，沿用H20/BF16/state。比对项目eager10步输出及各步VJP，另实际重算context并拷贝KV到图引用的稳定地址计时，固定shape不换context对象。
- 输出专家-only及含context预热5次median、准备耗时、内存；分别判断整H1秒及RTC 2d<=H约束，不声称窗口够就可用q8或任务成功。独立服务/guarded_run/共享锁，数值不符或RAM不足明确停。
- 已完成：FlashRT072f71c、PB449321b，M1/20261017T044731.565987Z-rtc-full-graph-59c3d2d8。raw max1.49e-8、VJPnormdiff0；专家582.23ms、context到raw879.38ms（各5median），普通FlashRT382.01ms（3median），准备8.59s。后者不含观测获取/snapshot/输出下载反归一化/控制，不称部署全链路E2E。d≈18/H20无合法RTC s，不跑任务。277tests（3defaultskip），专项CUDA5项实际通过，服务退出0。报告`../任务指导与相关参考/PerceptBudget_RTC完整10步生成测试_20261017.md`。不自动调H/N或扩大评测。

## 当前授权：FlashRT实验分支单步VJP（2026-10-17）

- 用户授权修改，但先做单次去噪前向+动作VJP gate。PerceptBudget及FlashRT-upstream已切experiment/rtc-vjp分支，既有frontend加载修复及tokenizer脏改保留，不混入新提交。
- 新上游rtc_action_vjp.py为显式启用CUDA Graph前向+VJP实验后端，先使用共享权重的可微expert，减少Python调度，默认无hook/无路径变化。不是已补齐native CUDA反向内核，不冒充完整RTC系统。
- 仅script/probe_rtc_step.py一个固定scene，检查eager/graph前向梯度、runtime target/mask、冻结权重、单步native对照、预热耗时。独立服务/guarded_run/GPU锁；通过单步后再报告决定，不自动跑完整任务或大编译。
- 单步已完成：FlashRT264e3f3新增入口、PB fb1b425探针；M1/20261017T044313.178469Z-rtc-step-graph-f1287b35。eager264.80ms/graph58.69ms（均5次median），四输出误差0，native单步max.0078125；target/mask与冻结权重检查通过，专项GPU toy4tests通过。服务退出0。scope仍shared-weight可微expert+graph，非native CUDA反向内核/完整RTC。报告`../任务指导与相关参考/PerceptBudget_FlashRT单步RTC梯度扩展_20261017.md`，不自动扩到任务。
- 全套272tests运行OK（2默认skip，其中专项GPU已单独4项实跑），新测试验证noise/target/mask/beta动态输入，不是固定旧结果重放。两仓库保留experiment/rtc-vjp分支；FlashRT默认源码与原有两项脏改未被本次覆盖。

## 当前授权：真正RTC去噪VJP接入探针（2026-10-17）

- 用户提供2506.07339v2并授权尝试原始RTC，不得用最终动作平均或前缀覆盖冒充。保持FlashRT文件不改；先尝试项目侧可微expert共享FlashRT已加载权重和context KV，只对noisy action求VJP、主模型权重冻结。
- 先单场景数值前向对照及一次真实梯度引导耗时/内存探针：原N10/H20/BF16/state不变，paper Eq2-5/算法1，beta5；同noise对照decode-only，relative RMSE<=2%仅作数值gate，不作成功率保证；未过gate不跑任务。真实延迟若已超过H可覆盖范围，不硬做q8任务。
- 新代码仅perceptbudget/rtc_expert.py、rtc_guidance.py和scripts/probe_rtc_guidance.py及必要测试，不重构正式运行器；独立run/systemd/guarded_run/共享GPU锁，不扩大评测。必要数值修正允许独立新探针，但不覆盖历史结果。
- 已完成核心接入gate：0885a62/f36029f，首次M1/20261017T042252.830728Z-rtc-vjp-gate-e75fd17b；预热M1/20261017T042416.054442Z-rtc-vjp-gate-c0e4fdd2。前向relativeRMSE raw.252%/前7维.196%，max.00390625；10个VJP均有限非零。warm expert-only2.578秒>H20名义1秒，不跑完整任务/不声称成功率或完整系统已接通。共享只读weights、FlashRT三关键文件hash未变。268 tests（1skip）、两服务退出0。报告`../任务指导与相关参考/PerceptBudget_原始RTC接入探针_20261017.md`。不自动投入编译/反向优化，不扩大评测。

## 当前授权：普通异步六维融合（2026-10-16）

- 在刚完成的无前缀普通异步论文融合基础，仅改前六维融合、夹爪直接取新预测，CLI --action-blend --blend-arm-only --q 8，task0一次。q8仍为boundary请求步间隔，不启用前缀、advance、减半或event小模型，BF16/N10/H20/state/seed7/init0/220上限不变。不改FlashRT或默认七维入口。
- 测试commit后独立systemd/guarded_run/共享GPU锁；全队列回放审核新夹爪来源、六维权重/过期索引及原H期限。仅单任务探索，不自动扩测或替换基线。
- 已完成：078bfcc，M1/20261016T125035.071631Z-action-blend-arm6-0d22e0e1，220步超时15.385秒，无busy/断供；28请求/27提交、140实际融合步。264 tests（1跳过），全队列与最新夹爪逐步审计通过，服务退出0。与七维初态/首预测相同但控制时序不同，不作严格单因素耗时归因。报告`../任务指导与相关参考/PerceptBudget_q8无前缀六维融合_20261016.md`，不自动扩测。

## 当前授权：无前缀的论文普通异步融合（2026-10-16）

- 用户确认测2609.22587论文普通异步+式8/9，不是原始RTC；本次去掉锁7步、提前锁定交接、六维特例及额外raw BF16融合舍入，直接用现成--action-blend --q 8，完整7维动作向量融合，beta1.2。task0一次，其他模型/初态/220步限制不变，不启用event小模型，不改FlashRT或默认基线。
- q8为请求boundary间隔，不是论文g（前次完成至下次启动间隔），也不等于此前advance-handoff目标交接周期；固定触发对照不声称完整动态策略复现。无前缀结果ready后立即按绝对控制步裁过期前缀并融合重叠段。
- 本地论文Related Work明确原始RTC为推理时梯度引导；本论文event模型训练需梯度，但式8/9推理融合不需要梯度。二者不能再混称。
- 测试、commit后独立服务、guarded_run及共享锁，仅一次；逐步公式/对齐/动作来源审计并保存报告，不因失败自动追加变体。
- 已完成：首run124104因bootstrap NumPy布尔判断错误invalid且未执行任务；bc9fba0修复并加回归。有效run=M1/20261016T124215.567055Z-action-blend-pilot-5d4ea659，220步超时14.722秒，28请求/27提交、间隔8×27、无busy/断供、138实际融合步；无prefix/advance、gain1全7维。260 tests（1跳过），独立全队列审计通过，服务退出0。报告`../任务指导与相关参考/PerceptBudget_q8论文普通异步融合_无前缀测试_20261016.md`；不自动扩测或替换基线。

## 当前授权：修正融合公式（2026-10-16）

- 用户要求修正：前次索引差恒定权重不是2609.22587式8/9。改为重叠段旧权重1到0，beta1.2；先裁剪两来源raw前六维再融合，消除裁剪顺序问题，保留BF16舍入供下一prefix exact。夹爪取最新、q8提前交接及默认不融合不变，不改FlashRT。
- 本轮先实现、测试和修正文档；修正后仅复核原范围task0/q8一次，不扩大GPU实验。旧失败数据保留并明确非论文公式，不能用于评价论文融合。审计工具兼容旧metadata以保留历史可复核性。
- 已完成：002bb65，259 tests（1跳过）；M1/20261016T122845.502911Z-advance-handoff-rtc-merge-a2e8c5c2同task0q8仍220步超时15.463秒，27次8步交接，无保持/普通断供；实际5点权重1/.629108/.354344/.150794/0，逐步审计通过，服务退出0。不替换不融合基线，不扩测。报告`../任务指导与相关参考/PerceptBudget_q8论文融合公式修正与复核_20261016.md`。

## 当前授权：q8提前交接加RTC式动作融合（2026-10-16）

- 在刚成功的advance-handoff q8/prefix7/late-hold上，仅加交接后旧队列与新预测尾部的六维RTC式两来源融合，夹爪取最新。task0一次、其余配置不变、独立run、不改FlashRT或默认基线，不称完整RTC。
- 旧/新按实际目标控制步对齐，权重exp(-0.1*abs(old_offset-new_offset))及1；融合raw32前六维后BF16舍入，反归一化执行并保留同一raw32给下一次前缀。已承诺前缀不改；不延长任何原H期限，停顿仍单列。
- 测试、commit后独立服务运行一次，核对实际承诺与融合命令一致、夹爪未融合、raw32来源及期限。与历史201步成功仅单任务对照，不宣布总体收益。
- 已完成唯一回合：60e31ab，M1/20261016T121438.943319Z-advance-handoff-rtc-merge-8b1c2e13，220步超时15.461秒；27个8步交接间隔、无迟到保持/普通断供，142融合目标/137实际融合步。257 tests（1跳过）、逐步公式/实际前缀/raw32审计通过，服务退出0。与未融合首态指纹一致但首预测raw有0.0053711差，不声称严格单因素。报告`../任务指导与相关参考/PerceptBudget_q8提前交接_RTC式动作融合_单任务测试_20261016.md`；不自动扩测或切默认。

## 当前授权：q8提前计算、定期交接小测（2026-10-16）

- 仅task0一次，q8表示目标交接间隔而非交接后再执行8步才请求；固定prefix7，在上次commit+1请求、+8交接。迟到零动作保持，下一周期以实际commit为基准，不并发推理、不延长origin+H。
- 显式--advance-handoff --prefix-lock --prefix-len 7 --late-policy hold --q 8，默认路径及FlashRT不改。BF16/N10/H20/state/seed7/init0、220步上限保持。独立run、guarded_run和共享GPU锁，测试commit后启动。报告实际交接间隔、暂停/断供、结果和时长，不称完整RTC。
- 已完成唯一回合：24027e2，M1/20261016T120654.285958Z-advance-handoff-pilot-3ffa5e8a，201步成功、14.181秒；交接间隔8×22/9×2，暂停2次各1步共0.11336秒，普通断供0。250项测试（1跳过），逐步实际审计通过，服务退出0。报告`../任务指导与相关参考/PerceptBudget_q8提前计算定期交接_单任务测试_20261016.md`。q语义较前q5改变，不能作纯q单因素结论；不自动扩测、不替换默认。

## 当前授权：q5 task0 prefix late-hold 小测（2026-10-16）

- 仅task0/q5一次，显式`--prefix-lock --prefix-len 7 --late-policy hold`；默认discard/8及正式基线不变，不修改FlashRT、不扩测，不称完整RTC。
- 提前ready等待锁定prefix结束；迟到每tick七维零命令，Panda夹爪0保持驱动目标（用户已独立核实）。ready后从锁定prefix之后接续，裁掉等待对应尾部，原origin+H20截止不变；raw32随接续队列保存以支持下次prefix。
- prefix不足或全部过期明确报错，保留请求、暂停次数/步数/墙钟；220步上限、BF16/N10/state/seed7/init0保持。测试commit后独立systemd用户服务、guarded_run及共享单GPU锁；启动前告知，无需再次审批。
- 已完成唯一回合：实现e3a89b6，run=M1/20261016T114424.164483Z-prefix-lock-pilot-0210f6a1，服务perceptbudget-prefix-hold-q5-20261016退出0。220步超时、15.356秒，暂停1次/1步/0.051579秒，普通断供0；153从索引7接续至原165截止，155的下一次raw32前缀通过。逐步离线审计通过。探针沿用boundary请求（实际5后21次10步），不冒充严格q5/latest_due。报告`../任务指导与相关参考/PerceptBudget_q5前缀锁定late-hold_单任务测试_20261016.md`；无遗留GPU作业，不自动扩测。

## q5不减幅衔接小测（2026-10-16，最高优先）

- 用户明确要求测试第一种替代：等待期正常执行旧动作、按实际进度接续，q=5。先task0正常场景一次，不扩扫描、不改默认基线。
- 显式--pending-arm-gain 1，仅将运动系数0.5改为1；request计q、progress接续、原H20期限、BF16/N10/state/seed7/init0与220步上限不变。独立run，不覆盖M1数据，不修改FlashRT。
- 结果如实报告成功/超时、步数、时间和断供；与历史0.5仅作单任务参考，不宣布总体成功率。运行前测试commit，独立服务，单GPU。
- 已完成：run=M1/20261016T113127.802223Z-q5-progress-gain1-17458302，服务perceptbudget-q5-gain1-20261016退出0；task0 q5 gain1为220步超时、14.99秒、断供0。220步全部gain1且实际/计划命令相同；与历史gain0.5的140步成功初态及首帧配对通过、首预测最大差0。报告任务指导与相关参考/PerceptBudget_q5不减幅进度接续_单任务测试_20261016.md。不切默认、不自动扩测。

## M1交付报告待用户审查（2026-10-16）

- 用户认可按实用口径汇总M1，明确不再追加严格验证。已拟任务指导与相关参考/PerceptBudget_M1阶段交付报告_审查稿_20261016.md，汇总配置、同步/异步结果、缓冲、阶段/扰动、功率和机会估计及数据入口；所有链接已检查。
- 本轮只整理文档，没有新实验、没有改计算口径；审查稿不等于用户已正式验收。下一步等待用户审查意见，不自动扩测、调频或推进M2。

## 当前M1.2功率测量（2026-10-16，最高优先）

- 用户明确不要求追加严格验证，1.1现有记录按实用口径交付；停止1.1扩测，不继续把B_safe证明当推进门槛。当前授权测1.2：复用异步150次日志计算推理/无推理功率、占空比和简化节能机会，另做一次短开关负载标定。
- 主agent唯一GPU短测：相同已加载模型、同一固定LIBERO场景，仿真/渲染/控制持续运行，交替不推理与连续全量推理，各8秒、三对后再8秒空闲，约一分钟。传感器20ms轮询仅用于观察响应，不冒充硬件50Hz采样。保留原始读数与时序。
- 后端/模型/精度/N/H/state和设备功耗模式、锁频设置不变，不改FlashRT，不运行动态调度，不重复150次。INA3221 VDD_IN只称模块输入功率。
- 结果写独立M1 run；旧q5/8为task加载、新q10/12/15为session复用的来源差异如实标注。S=(1-d)*(1-r)仅估计机会，不称实际节能；有断供的q15不当作推荐工作点。无需新增置信区间或严苛验收框架。
- 1.2已完成：短测run=M1/20261016T101130.349164Z-m12-power-calibration-6df1577e，服务perceptbudget-m12-power-20261016退出0；连续推理28.60W、无推理9.93W、r=0.347，2493样本无丢样。最终含标定分析=M1/20261016T101426.530042Z-m12-power-ee16be91，复用150回合；232项测试含真实日志只读集成通过。
- 报告：任务指导与相关参考/PerceptBudget_M1.2_功率与节能机会_20261016.md。短间隙分区与长窗口估计分开；q10/q12长窗口假设下机会约20.4%/28.0%，不是实际节能或还能额外省下的保证。实读功率过渡约1秒含传感器平滑，不当真实唤醒延迟。无在途GPU任务，不自动进入调频或M2。

## 已批准M1.1小规模补测（2026-10-16，最高优先）

- 用户明确批准测试阶段、可靠执行长度与一种固定外扰，要求与子agent协同。本次覆盖下方“仅离线/不自动GPU”的旧限制，但不授权重跑150次或改FlashRT。
- 范围固定task0/task3；同步q5/8/12、异步q8/12，各正常/扰动一次，共20回合，单模型、单GPU作业。BF16/N10/H20/state/双视角/seed7/init0、220步上限和既有异步衔接不变。
- 固定扰动在正式第30步动作前、采集本轮观测前，将显式目标黑碗世界x方向平移+0.02m，仅一次，保持姿态/速度、不推进仿真；强制更新观测并记录完整前后状态。称瞬时位置扰动，不冒充外力冲击。
- 同步补录只用于独立小测，不替换旧30/30。每回合保存完整预测、sim/物体/接触、帧和功率；逐条落盘、可恢复、独立服务。新运行前测试和commit，在途不改依赖。
- 阶段先用显式目标与证据保守标注，再人工核帧。两任务一次只能给局部探索结果，不能自动宣布普遍B_safe或1.1验收通过；未规定的轨迹容差不擅自设验收标准。
- 本轮run=M1/20261016T094730.333910Z-m11-stage-disturbance-1de2112f，核心a86cf7e，独立服务perceptbudget-m11-pilot-20261016，日志/tmp/perceptbudget-m11-pilot-20261016.log。20回合启动前相关全套218测通过，分析另5测通过。唯一GPU在途时不得改SOURCES中的依赖；analyze_m11_pilot.py纯离线可单独维护。
- 本轮现已20/20完成、服务退出0、自动分析complete_pilot；正常10/10成功，扰动task0为0/5、task3为5/5，均无断供。2958步逐条审计无重复/过期动作；同步prefix与扰动不变性均通过。全套223测通过。总结任务指导与相关参考/PerceptBudget_M1.1_阶段与扰动小测_20261016.md；核帧review_notes.md。局部成功覆盖到q12不等于普适B_safe，不称整个1.1验收通过；无新的GPU任务，不自动扩测。

## 当前质量对照：同步q5（2026-10-16，最高优先）

- 用户已将主要质量基线改为同步q5（30/30），不是异步q5。异步q5/8/10/12/15共150次均完成；最终历史汇总M1/20261016T082622.503351Z-async-final-edc1a3bc不覆盖。
- 本轮只做离线复用：同步原135+补15中的30条q5，对照异步150条；新建独立分析run，保存来源SHA和证据缺口，不修改原始记录、FlashRT或运行器，不自动重跑GPU。
- 150对首机器人raw_state完全一致；同步无首帧、sim指纹、完整预测和场景接触，不能宣称全场景严格初态验证或同步物理阶段已对齐。同步已有命令/轨迹可描述性比较；预测误差仍是异步预测与随后执行命令的一致性诊断。
- 剩余名义动作区间按H-elapsed扣除，progress偏移不一定等于elapsed；分开记录裁前缀、裁尾部及原有效期限。B_safe未经质量验证保持null。粗接触/运动标签不冒充平稳搬运或精细放置，外扰未测。
- 先交付已有证据与缺口清单，缺失记录补测需说明最小范围，不擅自扩150次或提前转功耗/动态策略。
- 已交付离线run=M1/20261016T084801.296732Z-sync-baseline-comparison-01cae795：analysis.json/md为同步对照，async_evidence.json/md为原始事件位置、命令误差、逐提交坐标核对及粗阶段/集合分项。194项测试通过，未跑GPU。总说明：任务指导与相关参考/PerceptBudget_M1.1_同步q5对照分析_20261016.md。状态partial，不将证据缺失写成全部完成。

## 断点恢复修复（2026-10-16，优先于下方旧进度）

- 剩余90次在task4初态校验停止：LIBERO硬重置会累积属性采样器、重新采样固定柜体；固定物体model位姿不包含在flat sim state中。实测柜体漂移约1.5cm，不能放宽图像校验冒充同场景。
- 项目侧固定首次正式场景的fixture布局；续测先重建并与已验证首帧核对。task4连续四次重置与原q10首帧及sim指纹完全一致。未修改FlashRT或安装的LIBERO。182项CPU测试通过。
- 同run显式initialization-amendment审计源变更并恢复；保留13条有效记录，仅重跑未通过校验的组合。每回合先保存pending结果，再校验和逐条更新总进度；真实初始化偏差、OOM等仍应停止，不隐瞒错误。
- 续测服务perceptbudget-remaining-resume-20261016，日志仍为/tmp/perceptbudget-remaining-20261016.log；原失败服务仅作历史记录。完成后自动汇总，运行中不改源码。

## 当前授权：完成剩余异步测试并汇总（2026-10-16，最高优先）

- 当前剩余run=M1/20261016T065722.043402Z-async-remaining-b80f7a6e，评测提交e7a7808；独立服务已启动并进入task0，q10/q12已各完成一回合。仅运行q10/12/15共90次，源码快照冻结；离线汇总脚本可并行完善但不得改动在途评测依赖。完成后自动调用compare_async_runs.py --require-complete并打印最终独立汇总目录。
- 离线汇总已完成验证：全163项CPU测试通过，真实历史q5/q8共30组初态/首帧配对通过（2组使用既有1级像素容差）；保留原分析limitations、源summary SHA与配置/生命周期差异，不改原始数据。仅离线新文件在GPU运行后提交，实际评测依赖未修改。
- 用户已批准将剩余q10/12/15各30任务全部测完，共90次；不重跑已完成q5/q8，不扩展提交后计步废弃实验。采用request计q、冻结的进度衔接gain0.5、BF16/N10/H20、双视角+state、seed7/init0及220/280/300步上限。模型session复用，原生环境清理，记录内存；不新增FlashRT改动。
- scripts/run_async_remaining.py串联唯一GPU评测和完成后的离线汇总；允许显式--resume同run，不重跑完成的任务。独立用户服务perceptbudget-remaining-20261016，日志/tmp/perceptbudget-remaining-20261016.log；外层guarded_run保护内存，不与别的GPU评测并发，在途不改评测源码。
- 离线compare_async_runs.py保留三个来源与SHA，核对固定配置/初态/帧并以q5配对，独立创建汇总run，不复制或改写原始trace。必须醒目标注旧q5/q8为task重载、新90次为session复用，真实耗时差异不直接全归因q；不能冒充同一轮执行器完全一致的150次。
- 初始原始来源：q5=M1/20261016T053343.959035Z-async-q5-baseline-a1f27927，q8=M1/20261016T035800.217970Z-async-q8-screen-3650b7d9。最终报告包括分suite/总体成功率、共同成功耗时、断供、真实刷新间隔及已记录轨迹/事件/缓冲分析；不将未过期区间自动称B_safe或声称M1节能验收完成。

## 用户恢复原异步基准（2026-10-16，最高优先）

- 用户决定仍以之前的异步规则为标准。config.refresh_anchor=request显式固定：按上次启动推理计q，模型空闲且间隔到期继续请求；保持已确认的latency_limited_progress衔接（等待期gain0.5、进度对齐、不延长预测期限）。这里“原版异步”不指撤销FlashRT现有两个补丁，也不指回到纯crop覆盖。
- 原q5/q8的30任务结果分别20/30、25/30，均无断供；继续作为历史基准。新实验保留模型单加载加速，生命周期变化需记录，不把历史耗时无差别拼接。
- 提交后计5步的9任务实验仅留作对照（8/9成功、78步断供），不作为基准，不继续扩测；原始记录不删除。不自动启动其他实验。

## 当前实验：提交后执行q5再推理（2026-10-16，最新）

- 本轮已正常完成：run=M1/20261016T064251.653092Z-q5-postcommit-nine-ea8a80dc，Spatial2/3、Object3/3、Goal3/3，共8/9成功；task1到220步超时。133次非bootstrap请求均独立核验为最近commit+5，且5步均正常gain1。减半比例59.0%，历史同9任务97.7%；旧同9任务7/9成功。新6任务合计78步断供，不可称质量和供给验收通过；不得延长有效期或隐瞒保持动作。
- 模型只加载一次、字段和自动分析完成，服务正常退出0。9个初态sim指纹与旧对应任务一致；旧task隔离、新session复用的执行器条件也不同，非严格单因素对照。仅实验，不自动切换历史默认、不擅改H或q。报告：任务指导与相关参考/PerceptBudget_q5提交后计步_9任务测试_20261016.md。
- 用户授权按其解释单测q5，共9不同任务一次：Spatial0/1/2、Object10/11/12、Goal20/21/22；模型仅加载一次。显式--refresh-anchor commit，bootstrap提交后执行5步再推理；每次新结果提交后重新计5步，不从上次请求启动时计数。其余BF16/N10/H20、双视角+state、seed7/init0、运动命令gain0.5仅在等待期生效、进度衔接和原有效截止不变，不加长步数上限。
- 新增post_commit策略为显式实验，不覆盖历史默认request/latest_due，不在旧run续测切换起点。每次请求记录anchor及anchor_step，非bootstrap强校验observation_step=latest_commit+q。单请求在途，不并发推理。
- 保留断供/超时真实记录，等待期不足不延长旧预测或静默缩短q；该规则可能使供给更紧，需要实测。独立run，不混入旧q5/q8总体统计。未新增FlashRT修改，暂不扩到其他q或30任务。

## 评测加速接入（2026-10-16，最新）

- 集成小测已完成：6任务全部有效结束（5成功、1超时），字段检查/自动分析通过、断供0，只加载一次19.78秒；相同6任务旧run累计加载105.71秒。清理后CUDA分配均6515.2MiB，后3任务RSS2601/2616/2569MiB，可用内存最低清理后约2718MiB。短测支持启用session默认，不证明30任务稳定；后续保留逐任务内存检查和显式task回退。没有新增FlashRT改动，历史q5/q8数据不覆盖。
- 用户明确要求减少每任务重载的等待；82e9a8b已将同进程模型复用接入scripts/run_async_q.py，config.model_lifetime=session。保留--model-lifetime task隔离回退；旧run未记录字段时仍恢复task，禁止resume切换生命周期。
- session模式保持同一衔接、模型精度/去噪/输入、逐任务预热、原始记录和GPU互斥锁。只改变模型/环境生命周期：保留原生渲染释放，按任务记录RSS/MemAvailable/CUDA内存，创建环境前至少1GiB可用；清理异常或内存不足明确终止，不静默改策略。同步入口不变，无新增FlashRT修改。
- 全149项CPU测试通过。正式入口集成验证为独立run=M1/20261016T062558.970535Z-session-reuse-smoke-e1ac05de，q8、task0/1/10/11/20/21共6次；服务perceptbudget-session-smoke-20261016，日志/tmp/perceptbudget-session-smoke-20261016.log。在途不改评测源码，不混入既有q8成功率统计。完成后核查一次加载、清理后内存、字段与自动分析，才能报告加速实测。

## 原版后端验证（2026-10-16，最新）

- 用户希望不修改FlashRT以降低维护负担，已授权隔离原版加载/短推理验证。原版e4fe123干净worktree=/tmp/perceptbudget-stock-fts501/FlashRT，复用相同CUDA源码编译扩展。当前正式config及补丁版未切换，无新增上游源码改动。
- 原版run=M1/20261016T061641.235413Z-state-latency-c3a3eb6b：加载期触发768MiB保护，最低MemAvailable609932KiB，退出125，未进入推理，不称内核OOM实证。已修正原run/summary残留running标记。对照补丁版M1/20261016T061746.474321Z-state-latency-9f059cd7正常完成，最低3583980KiB，加载17.43s，固定state367.40ms/变化state369.55ms。两者同一流程；未分别消融两补丁，不能声称tokenizer补丁必需。
- q5正式30次已完成20/30、零断供、分析完成。原生渲染释放的单加载6回合诊断也已完成、无CUDA报错，CUDA分配约6515MiB稳定，后3回合RSS约2414/2387/2388MiB；仅短测，不据此宣布30任务复用已修复或切换默认。
- 结论/数据见任务指导与相关参考/PerceptBudget_原版FlashRT加载验证_20261016.md。暂不直接撤销加载优化；不在未经明确安排时继续更改FlashRT或用运行时替换冒充原版。

## 加载复用排查（2026-10-16，用户要求减少反复加载）

- 不修改在途q5的任何评测依赖，不中断、不换算法。已静态确认eval_q._patch_egl_cleanup将EGL.free和两个析构器置空，旧LUTVLA相关入口未做该补丁；hard_reset也重建渲染上下文，故任务结束时加gc并不足以恢复完整释放链。尚未量化其内存贡献，不宣称唯一根因。
- 新增隔离诊断scripts/probe_model_reuse.py：同一模型测试task0/10/20各两次，保留原生渲染释放（不应用兼容补丁），逐任务记录RSS、可用内存、CUDA分配，保留完整原始记录。它不是新的正式基准，也不改上游。原兼容补丁旨在规避Jetson CUDA损坏，故未经GPU验证不能直接取消正式路径中的补丁。
- 独立用户服务perceptbudget-model-reuse-20261016等待当前q5 summary.completed后再抢GPU锁运行诊断；若q5异常或等待超时不运行。禁止与正式q5并发。诊断后审查CUDA/质量/内存，未通过则不能切换常驻模型默认。

## 当前推进：异步q5基线（2026-10-16）

- 用户批准继续M1，当前先补异步q5，30不同任务各一次；沿用已冻结latency_limited_progress、BF16/N10/H20、双视角+state、seed7/init0及suite上限220/280/300。已有q8的30次保留，不重跑、不拼接小测。q5实际请求间隔必须实测，不能将设置q5当作严格每5步刷新。
- 使用独立用户服务perceptbudget-q5-20261016，GPU顺序运行；完整记录预测、实际动作、场景、帧、功率、有效区间和断供。在途不改评测源码。之后的q10/12/15与功率测量按阶段推进，不由当前服务自动扩测。
- q5结束后与既有q8做离线配对分析，先核对初态、帧与固定配置，失败/超时完整保留；跨run配对不可伪装成同一run，来源必须记录。

## 异步衔接已确认（2026-10-16，最高优先）

- 用户已选定latency_limited_progress作为后续异步固定衔接方式，不再仅作为候选。新异步run默认读取config.action_handoff：latest_due、等待期间六维机械臂命令乘0.5、夹爪不变、按实际末端进度接续，保留H-elapsed步且不延长原截止。同步执行逻辑不变；旧run恢复仍严格遵从原快照，不切换方法。
- q8本轮30次已正常完成：Spatial10/10、Object7/10、Goal8/10，总25/30，5次超时、0断供。run=M1/20261016T035800.217970Z-async-q8-screen-3650b7d9，分析完成。此前进程随服务重启中断；从task2起由独立用户服务perceptbudget-q8-20261016续测，前两条保留。今后长GPU任务使用独立服务，核实进程而非只信running文件。
- 固定方法不等于达到质量目标：同步q8为30/30；共同成功25任务的wall均值同步12.320秒/异步11.708秒，执行步数120.56/202.32。后续如实保留质量与时长代价，不称完整RTC或已证明节能。未经新安排不自动开GPU扫描。

## 当前q8异步测试（2026-10-16，优先于历史）

- 用户授权测试当前候选衔接下q=8的异步成功率。固定profile=latency_limited_progress，仅q8，Spatial/Object/Goal各10任务、各一次，共30回合；seed7/init0、BF16/N10/H20、双视角+state及原功耗设置保持。
- 不补q5、不重启旧150次，不混合历史小测。每任务独立进程、GPU顺序运行，保存完整预测/执行/场景/帧/功率。单q无本轮q5配对，不能伪造相对q5轨迹结论。
- 入口scripts/run_async_q.py仅放宽单q入口，多q仍要求q5先行；模型和衔接计算不变。在途不修改评测源码。按总体及suite报告成功数、超时、实际请求间隔与断供，完成后才能给最终成功率。

## 最新修复状态（2026-10-15，以本节优先）

- 集成与最终复核均已结束，无在途GPU实验：087603b的task0/1/2×q5/8共6次全部成功；01152f6最终修正复核task1 q5/q8分别218/201步成功、14.307/13.585秒，无断供。两个run均完成自动分析，字段/初态/原H截止校验通过。小样本不代表总体成功率；保持候选显式启用，不自动切换默认或重启150次。
- 原150次停止。用户授权解决衔接问题，限小规模修复验证，不自行重启150次。
- 新可选profile `latency_limited_progress`：latest_due及时补发，推理在途时机械臂0–5维乘0.5、夹爪不变；按实际EEF位移选择新chunk偏移，裁掉相等数量尾段，剩余长度仍H减elapsed，绝不延长observation_step+H。不是RTC/前缀，不改FlashRT。原crop默认不变。
- 诊断task0 q5三次139/141/139步成功；task1一次220步成功、task2一次208步成功，均无断供；task1处于超时边界，不称总体质量通过。同样减速但纯时间裁切的task1/2均超时。实际q5多6–7步，控制约15–16Hz，不能宣称严格q5/20Hz。
- 087603b集成profile及对应事件时序/缓冲校验；项目venv全139项CPU测试通过。小规模正式入口集成run=M1/20261015T074553.575495Z-handoff-integration-814479b6，仅task0/1/2×q5/8共6回合。首次用系统Python因缺torch在模型加载前退出，改项目venv原run继续，无已完成回合被覆盖。在途不改评测源码。
- 失败方案代码保留于archive/async-handoff-investigation-20261015，各run独立源码快照保留。优先审查集成结果，再向用户报告；不得把小样本称95%成功率或混入旧数据。

## 当前范围与同步质量试分析（2026-10-15）

- 用户最新授权“解决动作衔接问题”：允许项目侧最小修复及小规模GPU对照，旧150次仍停止。当前新增scripts/probe_handoff.py，用同task0/q5/seed7/init0对照crop、原prefix、clean-prefix t0条件；主模型精度/N/H/感知及FlashRT源文件不变。发现原前缀逐轮写入干净动作但使用统一非零扩散时间；t0前缀是待验证修订，不以prefix_exact数值锁定宣称质量改善。
- 最新用户因异步成功率过低要求立即停止。已对当前父进程发送SIGTERM并确认全部run_async_q父子GPU进程退出，正式run停止于15个有效回合（task0–2各5q）；不自行重启或扩测。q5/8/10/12/15成功0/2/0/1/0（每q3次）；q15共69步断供。已确认q5多数实际间隔10步、每次结果裁掉约6–7步，task0前几个被裁预测前缀与实际旧动作不同；属于衔接假设待诊断，不能归为主模型固有低成功率。
- 离线查明昨日2/3来自旧45次30/45；同task0–2交集旧11/15、今3/15。旧eval_q超期后额外等待50ms（34a4315中next_deadline=now+period，01e799b后修正），task0 q5实测8.677Hz；今天probe约15Hz。推理均约0.4秒，期间旧3–4控制步/今6–7，q5旧按5步刷新/今多数10步。模型和输入适配从34a4315至今无变更。时间条件不一致已确认，不能把差异直接归为今日动作代码错误或纯模型质量；没有启动新GPU验证，保持停止。
- 最新用户已批准启动新的普通异步配对实验：30任务×q=5/8/10/12/15，各一次，共150次；以本轮异步q=5为动作/轨迹基线，保持BF16/N10/H20、双视角+state、seed7/init0及suite上限220/280/300。不同q必须核对初始仿真状态和首帧指纹。入口 `scripts/run_async_q.py`，每任务独立GPU进程且顺序运行，完整预测/逐步动作/场景接触/帧分别保留。无需补跑同步。
- 正式run=M1/20261015T062141.645304Z-async-q-paired-c491557a，评测提交1051976，108项测试通过。子进程继承GPU作业锁，父进程中断也不允许并行重复启动；每个q每次尝试独立目录。启动后不得修改在途评测代码；可读task summary/worker.log监控，完成后自动分析。独立preflight task0 q5超时/q8成功，配对指纹与字段通过，不能混入正式150次。
- 已在13次有效回合后因task2 q12首帧仅4个通道各差1级触发校验停止；sim/raw/object/contact完全一致。a78eaf7/0110bbb仅修正图像配对验证容差（max1、变化通道比例≤0.0001，sim state仍exact），23项相关测试通过；记录validation_amendments原新SHA后同run续跑，原始失败尝试不覆盖，推理/动作逻辑不变。状态以当前summary为准。
- 新异步分析记录配对成功率差与轨迹偏差95%区间（以任务为抽样单位）、夹爪/减速/转向的命令预测误差，以及提交时新预测扣除过期前缀后的剩余步数与秒数；旧队列剩余量另记。未过期动作不自动视为质量已验证的B_safe；阶段证据不足标明，外扰未测。两次短跑用于核验记录，不混入正式150次。
- 当前采用普通异步覆盖作为动作衔接方式：推理期间继续执行当前队列；新结果就绪后丢弃已过期动作，并以新预测替换剩余队列。RTC 融合仅保留为实验对照，不是默认路径。
- `probe_delay_buffer.py` 的 `--rtc-action-fusion` 已恢复为六维 RTC：按绝对动作步对齐 raw chunk，对机械臂 0–5 维使用 FlashRT 时序权重作归一化融合，夹爪维直接取最新预测；控制端实现，不改 FlashRT、不用模型 VJP。
- 先前同为 q=8、task0 的六维 RTC run 曾在214步成功，真实14.83秒、约14.81Hz，27次提交、143个重叠动作融合，融合CPU耗时合计约7.9ms，无忙时跳过/断供。该run原始目录已按此前清理要求删除；历史摘要可在提交 `e01df44` 的本文件中追溯。
- 为恢复实验记录于2026-10-15重跑一次：run=M1/20261015T044855.357558Z-rtc-action-fusion-6d-7d774eb8，task0、q=8，220步超时；真实15.14秒、约14.90Hz，27次提交、142个重叠动作融合，融合CPU耗时合计约7.89ms，无忙时跳过/断供。该单次结果与先前记录相反，不据此推断总体成功率；[重测报告](../任务指导与相关参考/PerceptBudget_RTC六维动作融合重测_20261015.md)。8项缓冲/公式单测通过。
- 用户要求在相同task0、q=8设置再测3次，2026-10-15已顺序完成：1/3成功、2/3到220步超时，控制约14.82–14.94Hz；三次均27次提交、143步发生融合，无忙时跳过/断供。仅为固定seed单任务观察结果，不当作总体成功率；详见[三次复测](../任务指导与相关参考/PerceptBudget_RTC六维融合q8三次复测_20261015.md)。

- 用户要求再试q=5，已完成一次task0动作融合运行，run=M1/20261015T035546.237156Z-action-blend-pilot-5dc8d1f0。220步超时、无断供，目标碗最大抬高0.129cm。推理均值397.9ms、控制15.64Hz；20/23个实际刷新间隔为10步，仅3个为5步，20个q=5刷新点因推理仍在运行而跳过。因此这次设置q=5但大部分实际更新节奏是10步，且是融合单次、无普通版配对，不估计成功率。详见[q5单次记录](../任务指导与相关参考/PerceptBudget_q5动作融合单次测试_20261015.md)。
- 诊断脚本新增 `--q` 参数以运行单个/多个指定刷新间隔；已保存本次运行源码快照。未改融合算法或FlashRT。

- 用户要求复核q8融合效果，已在同一task0、同一模型进程中做3组成对比较（普通裁切覆盖 vs β=1.2、7维全量融合），顺序交错；run=M1/20261015T034700.721579Z-q8-blend-ab-c9371f40。固定BF16/N10/H20/state/双视角、seed7/init0、40W锁频，普通版3/3于179步成功，融合版0/3跑满220步超时。三对初始sim state、首帧、state输入、首个H20输出逐值相同，第一段输出最大差0；融合队列更新发生后动作开始分叉。
- 融合版目标碗首次手指接触在126/178/178步，超过2cm抬起在191/201/201步；普通版均为第52步接触、第128步抬起。夹爪方向切换普通版每次9次，融合版19–23次。q8每个融合回合有26次非启动融合，N=5最多，反复作用于全程。平均推理392.7/393.1ms，融合更新约0.22–0.37ms，失败不由推理额外耗时解释。
- 算术公式和本次绝对步对齐未发现计算错误；但全7维连续插值把夹爪也当连续量，Robosuite Panda按夹爪动作符号开合，实测插值令方向切换明显增加。当前融合变体在task0表现明显退化；不能据一个任务推断全局成功率，也尚未分离夹爪插值与xyz平滑各自的影响。不自动修改融合策略或扩展到其他q。详见[配对报告](../任务指导与相关参考/PerceptBudget_q8动作融合配对对比_20261015.md)。
- `scripts/probe_delay_buffer.py` 新增 `--compare-action-blend-q8 [--pairs N]` 配对诊断入口，按AB/BA交错运行并保留每次请求、轨迹、帧和源码快照；本次实际源码快照位于run目录。实验和源码更新均保留，未修改FlashRT。

- 最新用户明确只试2609.22587v1 IV-C动作融合，不用事件判断器，不用前缀锁定/RTC；已完成run=M1/20261015T032201.233302Z-action-blend-pilot-5d8b012f，同task0、q8/12各一次。beta1.2，旧/新重叠按既有绝对观测步对齐；仅替换直接覆盖为Eq8/9指数融合，全部7维同权。论文未细化的索引对齐沿用本项目，称机制移植，不是完整复现；N1直接用新动作（公式只定义N>=2），N0无融合。
- 两次均220步超时、断供0、无busy跳过，间隔严格8/12，实际15.26/15.69Hz。q8融合26次，N5×21/N8×1/N6×2/N7×2；q12 17次，N1×11/N2×5/N3×1，实际中间权重过渡空间很短。目标碗最大升高0.783/0.465cm，均无超过2cm抬起，不能宣称质量通过。
- 权重和公式逐值校验、440步执行命令按事件序列回放核验通过；初态一致，每步仿真+.05秒，双视角各221帧，完整新旧动作/权重/实际输出/场景/功率保留，覆盖100%。融合更新平均CPU耗时0.301/0.123ms（本机含队列/记录构建，非单模型推理增量）。83项CPU测试通过，上游仍仅两个授权补丁，未扩测。
- q8切换处xyz命令变化均值0.0837（同chunk内部0.0995），但夹爪方向切换27次、抓取未成；不能把更平滑等同于更可靠。q12多数N1/2，不能拿该结果检验长窗口渐进融合收益。下一步需用户审查，不自动调beta、改夹爪处理或开150次。

- 最新用户批准仅改步速估计，不改迟到规则：最近16tick中位数改nearest-rank p20，其余配置相同；run=M1/20261015T030513.664746Z-adaptive-prefix-pilot-b325815c完成同任务q8/12各一次。q8仍迟到1次、断供0，在80步因队列4步不足所需7步停止；q12迟到0/断供0，但220步任务超时。实际14.90/15.59Hz，不能声称真20Hz或可靠衔接通过。
- 第二轮前缀：q8选7×5、8×3、6×1；q12选7×9、8×9。q8第72步按71.75ms历史较快tick锁6，结果第79步才被观察ready，晚于计划78；说明整个历史窗口较慢时p20也不能保证未来不变快。未进一步改迟到处理或自动调参扩测。
- 完整字段、raw及承诺动作核验通过，双视角81/221帧、功率全覆盖、初态一致、实际间隔8/12、无busy跳过，73项CPU测试通过。两轮文档统一保留在PerceptBudget_动态前缀小测_20261015.md，原run独立。

- 最新用户授权按预计推理耗时/实际步进耗时动态选择承诺前缀，同任务q8/12各一次。已完成run=M1/20261015T025805.412013Z-adaptive-prefix-pilot-176b7208；四种5/6/7/8图共用模型和缓冲，接口gate全通过。前缀选择在每次推理开始前确定，不在推理中途改变，不调q或功耗，不改FlashRT。
- 选择规则：最近16次服务耗时nearest-rank p80加5ms，除以最近16次实际tick中位数（至少50ms），向上取整，选满足q及队列上限的最小可用图；无解明确停止，不静默clamp。初始时延用任务外四图gate的中位数约380ms。
- 本轮未完成任务：q8在72步无可行前缀而停止，q12在96步同样停止，不是220步任务超时。q8选7×7次、5×1次，迟到丢弃1次、断供0；q12选7×4、6×2、8×1次，迟到1次、断供92–95共4步。实际请求间隔均8/12、无busy跳过。过往较慢步进使估计前缀过短，随后步进变快导致差1步错过交接，并因丢弃预测消耗剩余队列。不能宣称动态选择验证通过或零断供。
- 完整预测/raw前缀/选择理由/承诺和实际执行/场景/帧/功率均保存，输出前缀及实际承诺段逐值核验通过，初态一致，功率覆盖完整，72项CPU测试通过。未扩大重测；后续先审查耗时预测保守性与迟到处理，不自动调整参数重跑。

- 最新用户授权尝试RTC，若不能直接接入则使用现成前缀锁定，不改FlashRT。完整VJP方案只有provider协议无生产实现，外部重写VJP理论可行但不是开箱接入，本轮未重写；采用项目侧perceptbudget/prefix_model.py调用上游现成context/prefix图。
- 前缀小测已完成：M1/20261015T024551.720325Z-prefix-lock-pilot-648c42a3；同task0、q8/12各一次，固定prefix_len8，不是完整RTC。保留32维raw动作，逐轮前8步锁定exact；提前结果等前缀执行完才提交，迟到丢弃明确记录。本次所有已执行承诺动作逐值一致，无迟到/断供，实际请求间隔8/12；两次均220步超时，不能宣布衔接质量验收通过，不扩大评测。
- q8/q12推理均值约405/399ms（不含bootstrap）、实际控制14.85/15.14Hz，不能证明真实20Hz；计划交接时旧队列剩4/0步，区别于此前结果一就绪立即交接。完整预测/raw前缀/承诺区间/实际提交/场景/帧/功率均保留，source为运行时源码，67项CPU测试通过。run内旧uncommitted_prediction reason文字泛化不精确：结果可能提前完成但终止时承诺前缀未执行完，须依实际ready和scheduled_commit字段判断，不计为模型迟到。

- 用户后续批准先测同一任务、两个q各一次的真实延迟缓冲pilot。已完成task0、q8/12各一次，run=M1/20261015T020643.347200Z-delay-buffer-pilot-b9662ab9；q8成功171步，q12超时220步。实际触发间隔严格8/12，无busy跳过、无断供。真实控制速度14.38/15.71Hz，不能证明真20Hz不断供。
- 本次使用独立scripts/probe_delay_buffer.py，不改FlashRT、不启用RTC/融合、不调频。新chunk按观测步锚定、裁去已过期前缀并覆盖旧尾段，明确只是未验证衔接的诊断基线，不能当作解决了异步质量问题。
- 已保存每次完整H20×7预测与时序、归一化state、逐步动作前后raw state和物体观测、完整sim state、geom接触及距离、双视角原始帧（含终止帧）、动作来源/索引、频率/温度、完整功率。两次初始sim state一致，sim每步+.05秒、无重复过期动作执行，字段检查通过。q8提交时旧队列余量5/5/7步（min/median/max），q12为1/1/3步；这些是B_queue而非已验证B_safe，B_safe保留null。
- 64项CPU测试在开跑前通过。旧同步task0在q8/12均成功87/91步，本次异步质量变差仅是单任务诊断，不外推总体成功率。未扩大为150次，后续动作需根据该pilot限制审查。

- 当前以《PerceptBudget_任务书_简化版.md》修订后的M1范围为准，不提前开发动态刷新、调频、RTC或训练触发器。
- 用户批准先检查已有数据、分析两个任务，不直接重跑150次。已用原同步trace离线比较task0/task3的q5/8/15，共6条历史轨迹；结果在M1/20261015_sync_quality_pilot_01，脚本scripts/analyze_sync_quality.py。62项CPU测试通过，无新增GPU实验。
- 已有字段支持动作前末端轨迹、夹爪状态和执行命令；没有完整预测chunk、物体位姿、接触标记或视频。当前只能报告描述性轨迹差异及命令切换，不能声称测完预测误差、操作阶段或抓取失败原因。
- 路径形状采用累计路程归一化，仅为试分析，不是关键阶段对齐或标准真值误差。下一步对这两个任务补必要记录，保持同步固定配置，不混入异步或修改FlashRT。

以下保留为历史记录，若与本节冲突，以本节当前范围为准。

- 以完成当前 M1 实验为先，主 agent 负责拆分和审查；同一时间只运行一个 GPU 作业。实现与评测范围保持最小，不预建通用框架。
- 已实现最小异步评测、输入适配和功率采样，48项CPU测试通过。2026-10-14正式45次同机仿真初筛完成，q=5/8/10/12/15成功8/7/6/5/4次；无动作断供，无invalid，功率日志完整。实际仿真慢于20Hz，不能据此声称真实20Hz不断供或已完成低功耗标定；M1.2仅完成锁频环境的初步功率机会分析。
- 用户于2026-10-14要求重新开始M1.1：固定BF16、N=10、state、H=20、q=5/8/10/12/15；Spatial/Object/Goal各10个不同任务，共30任务×5q=150次，每组合一次，seed=7、init0、正式步上限统一220。全局ID0–9/10–19/20–29分别对应三个集合，记录局部ID和名称，分集合及总体汇总。旧45次仅为历史初筛，不混入150次。新增复杂任务可能受220步上限影响，超时不自动归因q。完成新M1.1后才继续M1.2。
- 新150次已启动，run目录M1/20261014T145920.598259Z-q-screen-dc24e13a；评测版本34a4315，55项CPU测试通过。子agent仅监控既有作业，完成后运行离线汇总；不重复启动GPU测试、不改在途代码、不调频，主agent审查最终结果。
- 开跑前核验 BF16、state 实际进入模型及 H=20 输出能力。H=20 不支持时记录原因并先确认后续目标；不改 FlashRT、不复制动作凑长度。state 规范未确认时不猜测预处理、不自行补维。
- 已核对 OpenPI pi05_libero 配置不使用 state，FlashRT state prompt 为额外输入实验。用户已批准本轮保持启用：按 checkpoint q01/q99 归一化8维state，不补维，结果注明，不自动切换。
- 第一阶段保持现场设备设置：目前 40W mode 4、CPU/GPU 已锁频。不得把现状描述成默认自动调频或擅自解锁。
- 每次运行标记异步执行或同步诊断。任务完成时间按正式控制步数乘固定控制周期计算；真实运行耗时独立计时并包含推理与等待。同步诊断暂停仿真的等待只计入真实运行耗时，不能冒充异步闭环结果。
- 实验结果写到 `/home/wyk/workspace/任务产出/perceptbudget` 的唯一 run 目录，不覆盖旧结果；保留配置、代码版本、任务、q、运行模式、状态和实际可用的时间/功耗记录。失败和超时也保留。
- 原版 FlashRT、旧 LUTVLA、共享依赖和现有虚拟环境保持不变。未经明确安排，不调整设备功率或频率设置。
- 用户已明确授权两个例外：加载内存修复（patches/flashrt_low_memory_load.patch）与分词器缓存（patches/flashrt_tokenizer_cache.patch）。只改变临时权重释放时机及 CPU 词表复用，不改模型计算、精度或 CUDA 源码。应称原版加已记录补丁；其余原版模块及旧项目保持不变。

## 当前重测（2026-10-15，取代前述在途计划）

- 旧150次已完成，但不是可靠的默认质量基准：忙时跳过刷新、异步裁切/覆盖动作以及Object/Goal统一220步混杂。q=8推荐已撤回，不把历史150次混入新结果。
- 用户批准先仅q=5同步重测，同一30个任务各一次。BF16、N=10、H=20、state、seed7/init0保持；仅修正执行流程和各suite标准上限Spatial220/Object280/Goal300。H=20/state仍为实验配置，不冒充官方默认H=10。
- 同步时执行完5步再推理，控制步暂停，等待只记入真实耗时；从新动作开头执行，不中途裁切。超期控制步不再额外等待整整50ms。同步能量不能当作异步节能效果。
- 小任务优先交子agent，主agent审查和启动唯一GPU作业；结果写独立q-sync run，记录模式、各任务上限、实际刷新间隔及断供。确认本轮基准后再考虑其他q与异步策略。
- q5同步30次已启动，run=M1/20261014T161425.228595Z-q-sync-2199aec2，评测提交01e799b，58项CPU测试通过。子agent监控现有作业，完成后仅生成质量汇总，不运行其他q或功率机会分析；主agent审查。

## 当前完整重测（取代仅q5安排）

- q5同步验证已完成30/30成功，实际间隔全5、expired0、underflow0，作为历史验证保留。
- 用户现要求重新测全5个q并交最终表格：q5/8/10/12/15，30不同任务×5q=150，全部同步质量模式，新轮含q5重跑，不拼接旧样本。H20/BF16/N10/state/seed7/init0保持，suite上限220/280/300。
- 同步等待只计入wall耗时，不计仿真任务时间，能量仅描述同步运行。交付总表与分suite成功数、共同成功时间对比、每成功能量、过期裁切和断供统计；旧无效150与q5验证都独立保留。
- 主agent启动唯一GPU作业并审查最终表格，子agent负责小范围文档和汇总实现。跑完150次才报告最终结果，不提前推荐大q。
- 本轮run=M1/20261014T163553.595415Z-q-sync-c47b9727，评测提交2cf363f，工作树启动时干净；五个q全部同步，尚未结束。离线汇总脚本可以完善，正在运行的eval/control/model不得修改。
- 完整重测现已完成：原135次因内存保护(684.4MiB<768MiB)在task27环境创建时安全终止，没有中途正式episode；剩余15次在M1/20261014T171602.439263Z-q-sync-supplement-ecd0339d按原配置补测。最终run=M1/20261014T172056.959144Z-q-sync-final-893d72a5为仅本轮135+15的汇总，来源和SHA保留，原数据未覆盖。q5/8/10/12/15成功30/30/29/28/25次；断供、过期裁切、推理异常均0，功率日志全覆盖，独立审查通过。59项CPU测试通过，不把同步测量当异步节能收益；后续先排查长进程内存压力，未自动推进其他策略或调频。
