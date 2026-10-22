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
