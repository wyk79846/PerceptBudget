# M2受限调频与恢复

当前阶段统一结果见[2026-10-22成本表与Fast配对复核](../../任务指导与相关参考/PerceptBudget_M2阶段成本表与Fast配对复核_20261022.md)。Full九项与真实Continue短测已完成；Fast旧KV交替的既有记录已补做整段能量复核，不重复GPU测试。完整M2验收仍有缺口，下文各“准备/尚未执行”描述是对应实施当时的历史记录，以统一汇总与AGENTS最新条目为准。

此工具只控制GPU、EMC、各CPU策略域频率，不切换40W功耗模式，不改电压、温控、governor、sudo配置或FlashRT。模型始终由普通用户运行。

## 当前状态

已完成模拟节点测试，**尚未执行真实硬件写入**。候选baseline/gpu816/emc2133/cpu1190分别用于参照及单因素筛选，不是已经选出的三个最终组合。GPU/EMC单位Hz，CPU单位kHz。只允许候选白名单，EMC同时检查DVFS表及当前合法范围。

每次lease最长600秒；保存原GPU/CPU min/max及EMC rate/lock，设置后读回控制值，并打印当时实际频率。READY只表示节点设置成功，不能当硬件稳定时间；正式测量仍需连续记录实际频率。工具不执行模型、不提供shell命令接口、不接受任意sysfs路径。

## 将来执行时的命令

先检查（不调频）：

```bash
cd /home/wyk/workspace/PerceptBudget
sudo /usr/bin/python3 scripts/control_device_clocks.py inspect
```

实际调频首次只做短时硬件验证，另行确认后执行，例如GPU816MHz保持15秒：

```bash
sudo /usr/bin/python3 scripts/control_device_clocks.py lease --profile gpu816 --seconds 15
```

看到READY后进入窗口；结束显示RESTORED。正常退出、设置中途失败、Ctrl+C、TERM、HUP均尝试恢复。恢复会逐域尝试并检查读回；失败以非零退出并保留快照，不能只凭进程退出认定已恢复。

若进程被SIGKILL、崩溃或没有打印RESTORED，确认原工具已退出后运行：

```bash
cd /home/wyk/workspace/PerceptBudget
sudo /usr/bin/python3 scripts/control_device_clocks.py restore
sudo /usr/bin/python3 scripts/control_device_clocks.py inspect
```

快照固定在root所有、0700的`/run/perceptbudget-clocks/original.json`，只在成功恢复后删除。存在未恢复快照时禁止新lease，锁禁止两个工具同时修改。restore不接受外部快照路径。原lease还活着时restore会拒绝占锁，应先正常终止lease，不要强行绕锁。`/run`重启会清空，跨重启应重新检查系统实际设置，不能依赖旧快照。

## 限制

- 暂未与模型实验入口联动；计时窗口、输出采集、GPU占用协调仍需下一步接入。不能直接把15秒短测当M2成本数据。
- 不授权长期后台代理，不开放sysfs写权限，不新增免密sudo规则。此项目脚本本身属于用户可修改代码，显式sudo执行是用户授权操作，**不能把它当作抵御恶意用户的提权安全边界**。
- SIGKILL与断电无法保证finally恢复；保留独立恢复命令。其他程序并发改频会影响结果，实测前应排除。温控仍可使实际频率低于请求值，不关闭保护来强制一致。

## 四候选常驻模型短测（2026-10-22准备）

用户已授权四组各5次完整刷新及保持短测。GPU816MHz硬件短测已由用户执行：READY实际816MHz、RESTORED后GPU1173MHz/CPU1497.6MHz/EMC3199MHz与原设置一致。CPU/EMC变频尚未硬件验证。

新增`profile_clock_screen.py`，仅普通用户启动，先sudo -v授权，再由同终端会话工作进程加载一次Full模型、原RTC专家与精确长度图。四组均H50/N5/state/双图、不压缩、不复用旧KV，输入来自既有成功Spatial0记录、同旧动作和d5/s5，同种子列表。只对固定输入做RTC完整计算，不执行LIBERO。

每组root工具锁频25秒并每0.2秒打印实际频率，由普通用户落盘；0.5秒等待、2次未计时预热、5次计时计算、0.5秒间隔、3秒模型常驻静止保持，之后等待自动恢复才进入下一组。每组保存动作、差异、频率trace、原始功率/覆盖率，缺覆盖不填完整能量。另存加载/准备成本。首次加载和切换均不混入稳态计算；静止保持无环境/相机/控制负载，不能冒充Continue板级保持功率，单批5次不能支持尾部或最终组合结论。

执行命令（不要在整条命令前加sudo，不用systemd-run）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py
```

启动时输入一次sudo密码。模型必须保留普通用户身份，root子工具不接受模型命令。工作进程保留同终端会话，避免新会话丢失sudo的终端凭据；加载前再检查sudo -n。父进程复用768MiB保护阈值，每0.1秒检查，异常只停止自己启动的普通用户工作进程，不向整个进程组发信号。若内存保护/强杀工作进程，root工具可能继续至本次25秒限时然后恢复；确认其退出并检查日志，若未确认恢复则使用独立restore。不得立刻重跑或宣称已恢复。

输出集中在`任务产出/perceptbudget/M2/<时间戳>-clock-screen/`，终端打印run_dir，进度见其中worker.log。sudo授权失效/调频失败即停止，保留失败记录，不自动重载模型续跑，不跳过不支持的频率凑齐四组。前期代码无真实GPU/调频试验；需要用户执行后读取结果。

## 已完成四候选短测与下一步

2026-10-22用户完成`M2/20261022T054411.150954Z-clock-screen-51594b24`：模型加载1次，四组各5次，实际频率采样均匹配，动作误差全0，全部恢复；四组计算/保持功率积分覆盖均1.0、丢样0。

| 配置 | 中位计算耗时ms | 每次计算平均J | 静止保持W |
|---|---:|---:|---:|
| baseline | 478.06 | 12.6765 | 8.7654 |
| gpu816 | 631.94 | 12.2231 | 7.5148 |
| emc2133 | 616.48 | 13.1682 | 7.3899 |
| cpu1190 | 479.69 | 12.6632 | 8.7229 |

仅单批固定顺序，初始CPU温度约54℃降至51℃，不能把全部差异唯一归因频率或据此选定最终组合。用户同意继续组合保持与切换成本小测。

### 组合保持与切换短测

同一入口新增`--transitions`。模型仍只加载一次，随后不做新推理，保持权重与图常驻。固定顺序：高频参照pre、GPU816+EMC2133/CPU1497.6、GPU816+EMC2133+CPU1190.4、高频参照post。每组5次循环，每次低频保持3秒→高频保持3秒，前后参照不真的降频；前后参照暴露漂移。root控制器固定此有限流程，20ms频率读回，普通用户同频率采功率；保存原始trace/功率、起止温度和恢复记录。每个root子流程约30秒，最大正常匹配等待约5秒，调用端等候上限55秒；不是无限常驻代理。原四组入口维持原列表不自动增加样本。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --transitions
```

每次记录调频命令、设置读回、首次观察到实际频率匹配的时间。读回是软件可见的延迟上界，**不是GPU首算子或新动作可提交的恢复时间**；没有在本次静止流程伪造这两个时刻，后续设备恢复服务测量仍待补齐。

保持功率使用每相位稳定中段（匹配后0.5秒至下一切换前0.2秒）。完整循环能量始终包含两次转换，不能只取稳定段推算整个能量。估计循环额外能量：完整循环能量减去按同等时长、实测高低频稳定功率计算的理想保持能量。高频参考按pre/post插值；漂移与五次循环波动构成实用噪声检查，不冒充正式置信区间。残差为负或未超过噪声标为unresolved、不填回本时间；可分辨正残差时才给平均往返能量/功率差的初步边界。不是分别测出的升/降频能量，也不是单次尾部分布。

本次仍非真实Continue控制负载，不能直接宣称闭环节能或完成27项表。原基准输入/配置和FlashRT不变，未开始真实执行；用户运行后再分析。

### 已完成组合切换与下一步恢复推理

用户完成`M2/20261022T055923.161737Z-clock-transitions-87883481`：单模型、四组各5循环、全恢复、实际频率匹配、功率覆盖1/丢样0。基准静止约8W，GPU+EMC降频5.999W，另降CPU5.943W；下/上频读回中位17.32/18.06ms与18.45/18.77ms。额外往返能量残差为负（约−0.36/−0.38J），不能解释为切换产生能量，均标unresolved，不填回本时间。暂优先CPU保持1497.6MHz，继续研究GPU+EMC低频等待，不据此选定最终三个组合。

恢复推理沿用同一入口：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --recovery
```

模型加载1次，H50/N5/state/双视角/原RTC不变。六组各5次，共30次固定输入请求：高频无额外等待pre、高频等0.2秒、GPU+EMC低频等0.2秒、高频等1秒、GPU+EMC低频等1秒、高频无额外等待post。0.2/1秒仅是本轮代表低频驻留时长，不声称对应硬件睡眠阈值；无额外等待组的进程/通信间隙不伪称零空闲。

root工具只接受固定五次RUN/DONE握手，不能执行用户命令，恢复准备好后才通知普通用户算动作；推理结束才进入下一次空闲，不让后台定时器提前切换。每次握手最多3秒、空闲白名单0/0.2/1秒、频率匹配最多0.5秒；异常/管道关闭即报错并走恢复，不常驻无限等待。正常流程上限预算40秒/子流程，调用端等待上限55秒；内存保护停止普通用户工作进程后，管道EOF会触发root恢复，仍需核查RESTORED。

每个请求保存：低频准备命令/匹配、实际低频驻留、升频命令/匹配、predict入口、动作返回；同时保存自上一次推理结束至此次predict的实际总空闲（含进程/IPC/种子/准备开销），第一条未知写null。初始观测固定；不在空闲中偷偷生成新动作。

两种能量窗口分别为：升频开始→动作算完（恢复+计算），低频准备开始→动作算完（完整等待循环，包含入低频成本）。同种子配对保存动作误差；检查空闲阶段实际频率（非零驻留）与推理阶段高频，功率不足全覆盖直接停止，CPU/温控保护保留。

**这是固定输入恢复+计算成本，不是完整实时服务L**：包括受限工具与普通用户协调耗时，但不含新相机/环境观测获取；predict入口不冒充GPU首算子时间。不能据30次诊断声称成功率/闭环节能/27项表完成。结果在新`clock-recovery`目录，不改写两轮既有证据；需要用户终端执行后分析。

### 已完成恢复推理与Full九项成本初表

`M2/20261022T061318.153103Z-clock-recovery-c914c106`已由用户执行完成：30请求、单模型、动作误差全0、功率完整覆盖、空闲/服务频率全匹配、六组全恢复。等0.2秒高频/低频恢复+计算451.31/470.47ms，完整等待循环13.7739/13.6299J；等1秒450.88/466.20ms，循环20.4788/18.6192J。观察到完整循环能量约省1.0%/9.1%，相应总循环增加约35.5/32.3ms；不能把单批诊断当闭环节能证明。功率读数存在响应延迟，短恢复窗口能量尤其不能直接当纯算子能量，尾段残留及后续读数响应尚需后续检查。

下一轮仍沿用原入口，不新增工程分支或改FlashRT：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --matrix
```

单模型、Full双图H50/N5/state、9项各5请求共45请求。三个频率组合是**待比较候选**，不是提前认定的最终点：

| 候选 | GPU MHz | EMC MHz | CPU MHz |
|---|---:|---:|---:|
| baseline（快速参照） | 1173 | 3199 | 1497.6 |
| balanced（新增中间候选） | 1020 | 2750 | 1497.6 |
| gpu_emc（低频候选） | 816 | 2133 | 1497.6 |

每个组合对应无额外等待、短驻留0.2秒、长驻留1秒：无额外等待组已处于本次服务频率；短/长组先在gpu_emc低频驻留，再切到对应服务频率。服务目标为gpu_emc时不会假造升频，仍记录无切换的准备/读回成本。真实空闲间隙与首个请求包含的进程开销继续逐条记录。首次组合准备、加载/建图不混入每次服务。

原受限recovery协议增加白名单serve-profile参数，仍五次RUN/DONE、固定idle白名单、正常异常恢复，频率合法表在首次写入前检查；原--recovery/--transitions/默认四候选入口不扩样本或改变原比较。

输出`clock-matrix`目录：`clock_matrix.json`及`full_cost_table.csv`九行，列出目标频率、起始空闲频率、耗时中位/样本最大、恢复+计算能量、完整循环能量、动作差异和恢复状态。5个样本只作短测，不报可信P95/P99。三点若存在明显被支配点再替换，不为凑数定为最终组合。

这仅是Full固定输入成本初表，不是27项验收表：Fast混合可用性、真实Continue控制负载、在线新观测完整服务及入选点追加批次仍未完成，不伪造其他两档数字。尚未执行本轮硬件测试，需要用户终端执行后分析。

### 九项首次失败与EMC控制修正

`M2/20261022T064312.512940Z-clock-matrix-331ded3a`无效退出：baseline_hot五次已完成，balanced_hot在root设置阶段报EMC读回不匹配，尚未推理；并非任务失败或RTC数值错误。balanced日志有RESTORED，旧恢复过程读回通过后才打印，因此设置失败但恢复成功；原父层将非零退出误报为恢复未确认，现已区分。

检查本机NVIDIA `/usr/bin/jetson_clocks:586`，其顺序是先写mrq_rate_locked=1，再写EMC rate。旧项目代码先解锁、设rate、再锁，存在自动DVFS争抢窗口，已修正为先锁再设，恢复也先锁再设原rate，最后恢复原锁状态。不新增BWMGR/电压/温控控制，不改合法频率或功耗模式。EMC精确读回允许最多0.5秒，超时报告requested/actual/lock；不放宽频率匹配、不把取整当通过。旧日志未记录实际错值，尚不能断言竞态就是本次唯一根因；需再次硬件确认。

原--matrix命令现在先对三候选各做1秒独立lease并恢复，日志写各profile_preflight.log，任何失败会在加载模型前停止（NO MODEL LOADED）。通过才启动单次常驻模型九项流程；若仍失败保留真实读回与恢复信息，不自动改频率凑数。模拟自动DVFS回退、错值、恢复成功但任务非零及预检提前停止均有测试。未执行真实修正后调频。

### 中间候选更新：2750MHz不生效，保留原EMC

2026-10-22用户复验`6ca5aaad`、`74620292`均在预检停止，未加载模型；baseline通过，balanced请求EMC2750000000、实际3199000000、locked1，持续0.5秒不匹配。两个日志均RESTORED，完整结果保留原run，不将预检失败算任务失败。修正锁顺序没有解决该挡位问题，不能再认定是读回竞态或取整，也不能宣称硬件永远不支持2750MHz；只确认当前受限接口设置未生效。未扩展控制到BWMGR、驱动、功耗模式或其他节点。

为了继续筛选且不反复加载模型，明确更换候选（不是静默把请求2750当3199）：

| 当前候选 | GPU MHz | EMC MHz | CPU MHz |
|---|---:|---:|---:|
| baseline | 1173 | 3199 | 1497.6 |
| balanced（更新） | 1020 | 3199 | 1497.6 |
| gpu_emc | 816 | 2133 | 1497.6 |

九项表规模、空闲时长及样本数不变。EMC3199已在真实测试中通过，GPU1020仍须模型前预检；新候选尚未实测或认定优于其他点。继续运行原--matrix命令，实际目标配置会落入新run的candidate_clock_profiles与root日志，不覆盖旧目标/原始证据。所有数值读回仍精确匹配，不开放容差。

### Full九项已完成，继续补真实控制负载

用户完成`M2/20261022T065643.580976Z-clock-matrix-2cd04e4e`：九项45请求、1model、三个预检/九组恢复通过、实际频率全匹配、功率覆盖1/丢样0、动作差异全0，CSV已生成。按无额外等待/低频等0.2秒/低频等1秒，baseline恢复+计算463.24/476.90/478.39ms，balanced496.18/514.12/513.47ms，gpu_emc650.58/651.96/653.61ms。热运行循环能量11.3243/11.4439/11.1460J；中间候选未展示明确优势，三点仍候选，Full仅固定输入初表。

下一步命令：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --continue-control
```

模型/专家各加载一次，使用已验证硬重置/共享EGL缓冲释放，不改安装包或FlashRT，保留900/768MiB保护。只构造一个Spatial0环境；三个频率各5次，共15个窗口、最多75个正式控制步，不是15条完整任务或成功率评测。

每次在高频下硬重置seed7/init0、记录同初始物理/双图哈希，基于**当前真实观测**重新bootstrap生成chunk；之后受限root工具仅保持目标频率2秒，普通用户执行该chunk首5步，不在窗口内再做推理。每一步仍包含原双路渲染env.step、scene/state/contact采集、图像拼接、success检查、trace和名义50ms控制节拍，记录实际Hz、envstep耗时、超期次数、实际频率和瞬态功率；终止就提前停止，绝不补零、重播或执行第6步。

保存观测时刻/step0、预测就绪时刻、承诺索引0..4、名义预计执行区间、逐步来源/执行时刻，H50未执行尾段不入缓冲。0.25秒只是5×0.05秒的名义覆盖，不当作已验证的真实墙钟B_safe。每次窗口后离线等待root恢复/环境重置属于测量准备，不伪称仍在执行有效旧动作。准备/重置/模型推理/写压缩视频数据不算Continue窗口内的控制服务。

功率采样从本次bootstrap前持续至限时root恢复后，保留完整功率原始记录；主要窗口积分仅是这5个动作的实际控制时段。因为紧接首次Full bootstrap且窗口约0.25秒，存在推理后器件/传感器过渡，称**瞬态Continue功率**，不直接替换稳态P_idle。该小测是初始chunk的真实控制负载检查，不是RTC交接、接触/精细阶段、九种Continue状态或完整M2验收；不训练、不延长动作尾段制造长期保持窗口。

输出在`continue-control`新目录：continue_control.json、逐窗口trace/原始功率/双图帧/频率记录。15个初始物理/图像哈希和首5动作必须配对一致；记录终态是否一致，不挑成功重测。仍先预检三个频率，无效则不加载模型。尚未实际执行本轮，由用户终端运行后分析。

## 2026-10-23：Full服务边界短测（M2计划4.5）

用户暂缓空闲时长扫描，优先补Full服务边界，不优化混合调度或进入M3。沿用原入口：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --full-service
```

只测高频baseline、5次服务、无额外等待，模型和原RTC专家常驻一次，先1秒频率预检。H50/N5/state/精度/RTC求解/40W不变，不修改FlashRT或安装包。

### 本次实际计入什么

起点：现有受限root工具SERVE事件的设备准备命令时刻。确认实际频率后，普通用户侧调用本机robosuite `_get_observations(force_update=True)`，强制重新运行双摄像头渲染和state传感器；不是复用准备好的图片，也不以零动作env.step获取画面。已检查安装源码base.py323和robot_env.py403，force会调用camera sensor重新render。获取前后比较sim.get_state，保证未推进物理。

随后调用原`_make_snapshot`旋转/缩放两图并归一化state，原RTC adapter重算完整上下文与动作，再用`probe_delay_buffer.commit`完成队列裁剪及提交包准备。终点是提交包内动作数据准备完毕；保存分段计时和起点至终点完整能量，整体分位数不由分段相加。

### 观测、旧动作与时间对齐

先硬重置Spatial0 seed7/init0，基于实际观测bootstrap产生初始chunk，再按顺序执行其前5步。这是一次性准备，不混入服务计时。余下45步原始动作是RTC参考，不读取另一个任务的旧动作。真实step5观测先用于离线准备对应精确前缀图；旧准备图释放，复用同一专家/建图流，验证raw/VJP图对照，之后不在测量中临时建图。

每次服务都在设备准备后重新读取step5的新观测。消费线程在这个设备成本诊断中暂停，实际控制步推进为0；d5/s5只是原RTC条件输入，不能声称实际推理消耗了5步或当成真实异步成功率。队列提交仍使用原生产函数，origin=5、consumer=5，名义输出区间[5,55)，不把起点改成到达时间来重新增加H。提交在隔离队列副本上准备，不注入实际机器人消费者，不更改主路径衔接策略。

每次保存观测开始/完成、预处理开始/完成、推理开始/完成、提交准备/就绪、图像及物理哈希、raw state、RTC记录、提交包动作及绝对索引、观测到提交的年龄、功率和频率。CPU观测入口不冒充GPU首算子；该时间戳仍写null，作为后续trace缺口。不把名义H50尾段标成B_safe。

### 结论范围与安全

这是**静止仿真状态上的新观测Full服务成本**，包含额外的独立双路重新渲染，和此前“已准备输入”的计算时延分表；不冒充生产中从控制步直接取现成最新观测的耗时，也不是控制/推理竞争下的异步闭环、物理相机传输或真机服务分布。先检查本次边界和数据完整性，通过后再决定对哪些有效条件追加测量，不扩展空闲扫描或Fast插入比例。

继续保留900/768MiB保护、已验证硬重置渲染释放gate和root自动/独立恢复。原结果不改写。输出到新`full-service`目录的full_service.json、worker.log、逐请求动作提交包、频率和原始功率。五个请求只报中位/最大和平均能量，不报可信P95/P99。本轮尚未执行，需要用户终端运行后分析。
# Fast旧KV专家重算三档短测（2026-10-23）

双视角token压缩候选成本补测：`--pooled-clocks`，原生factor2池化，两个256-token视觉网格各变64，总encoder视觉token128；SigLIP视觉骨干仍完整。每档5次/单model、layout检查、不改RTC/权重/安装包，不代表质量通过或完整服务。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --pooled-clocks
```

后续单腕部视角成本补测：同一入口使用`--visual-clocks`，三档各5次、一个单视角模型。真实减少为1视角/256视觉token并重新上下文，不复用旧KV；现有质量风险不因成本测量而消除。固定输入计算成本，不是实时新观测完整服务。命令：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --visual-clocks
```

普通终端执行，需要sudo授权，模型普通用户运行且只加载一次：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --fast-clocks
```

三档各5次。每个样本先在固定输入上重建Full上下文（不计Fast耗时），然后仅复用一次KV、重算完整RTC专家。旧KV包括旧state，非新感知刷新；记录名义age5，不声称实际场景变化质量。预检/频率读回/恢复/内存保护沿用原机制。不修改FlashRT或生产配置。

新run保存clock_screen.json及分档原功率/动作/恢复记录。每请求energy才是Fast窗口积分；整批compute含Full准备，不能除以5当Fast能耗。INA响应可能混入Full尾迹，窗口能量是诊断值，不能当独立专家能量或直接与实时观测Full服务计算节能率。其他Fast候选另行测量。

# Full实时观测服务三档同轮补测（2026-10-23）

高频单档已完成：476.703ms、11.499J。下一轮各档5次，共15次；一个常驻模型和环境，配对噪声、图像、state和输出动作。测量范围仍是冻结物理step5时的设备准备→实时观测→RTC→提交包就绪，不是异步任务成功率。没有新增空闲时长扫描，不修改FlashRT或生产基线。

普通终端执行（需要sudo验证；不要用systemd-run）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/profile_clock_screen.py --full-service --service-profiles baseline balanced gpu_emc
```

频率先预检再加载模型，每档完成后恢复。结果写入新M2 run的full_service.json、分档功率文件和summary.json；失败时保留证据，不把未完成数据算作有效成本表。

