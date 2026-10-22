# M2受限调频与恢复

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
