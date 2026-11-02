# 三档频率尝试

## 原方案保留

- Git分支：baseline/full-two-frequency-20261102。
- 基线提交：bbe414fc42f15d282309d5c5574f4ecbff86253a。
- 原高频Full：GPU1173/EMC3199/CPU1497.6MHz；原低频Full和Continue：GPU816/EMC2133/CPU1497.6MHz。
- 机制：shared_deadline_v4、25ms余量、原关键等待及阶段刷新规则，无Fast。历史12任务两组各10/12，逐任务及能量见M4_FULL_FREQUENCY_12TASK_RESULTS_20261102.md。
- 配置和关键代码SHA256清单：assets/full_two_frequency_baseline_20261102.json。已有模型、代码和原始实验不删除，不覆盖。工作区周报改动不属于此基线，不修改它。

原方案复现入口（先短测task0）：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4.py \
  --acceptance-group m3 --low-full --task-indices 0 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

## 新方案与推进顺序

1. 高频1173/3199、中频816/2133负责Full；低频306/2133只负责Continue。CPU保持1497.6MHz。新方案独立接入，原low-full开关和默认行为保留。
2. 先补测306↔816MHz的切换时间。已有低频306→完整高频约17.98ms，不能直接当作低→中频成本；既有中频Full样本也不能冒充从306MHz恢复后的完整服务实测。
3. 按原有关键动作与缓冲边界给出必须完成时间，比较中/高频的恢复＋服务P95＋25ms余量。普通阶段中频赶得上用中频，否则高频；关键阶段沿用原高频保护，先不增加新判断规则。高频也赶不上则立即启动，并沿用现有关键等待。进入低频前安排恢复时间，不等到截止才判断。
4. 等待段的降频收益必须覆盖切换成本；首版沿用已有最短等待规则，并记录实际切换及能量，不宣称已重新校准最优等待阈值。正在计算时不降到306MHz。RTC前缀与高/中频延迟历史分开保留，不能混用。
5. 接线后先做原方案与三档方案同任务配对，完整记录等待、恢复服务、任务耗时/能量、实际频率及成功。通过短测再扩展原12任务；不靠单任务宣布达到20%节能。

当前状态：基线保存完成；306↔816MHz测试入口已就绪；三档闭环尚未接入或运行，不影响原方案。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/probe_gpu_clock_switch.py --medium
```

普通用户启动，按提示输入sudo；10次往返，不加载模型，EMC/CPU固定，结束恢复。结果在新的M4/*-gpu-clock-switch-*目录。命令到频率读回不等于模型恢复至动作提交的完整服务时间。

## 低中档切换实测完成

运行M4/20261102T122709.206402Z-gpu-clock-switch-6a72818c。固定EMC2133/CPU1497.6，10次往返：816→306MHz中位2.919ms，最大3.052ms；306→816MHz中位2.840ms，最大3.131ms。20次实际频率及非GPU设置检查通过，RESTORED确认，原始日志重算与summary一致。该数据是静态设置至读回，不包括模型驻留恢复/推理/提交轮询，不能直接当作闭环完整L。至此独立低→中频转换约3ms，低→完整高频仍参考既有18ms整套设置。三档闭环尚未接线，旧基线不变。
