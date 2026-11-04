# 四组LIBERO全任务对照

Spatial、Object、Goal、LIBERO-10各task0–9，每任务一次，seed7/init0。固定q30异步与当前动态调度各40回合，共80回合。两组均启用仅请求时采最新双图，保留完整双视角、state、H50、N5、BF16与原RTC。动态组沿用中频Full优先/高频回退、306MHz Continue、50ms空档门槛、25ms启动点与余量、关键等待/阶段刷新和首次刷新30步。项目内部组名`m3`表示当前从M3迭代的Full＋Continue路线，本次实验属于M4扩测。

旧12任务清单与结果保留；`--all-tasks`显式生成新的40任务清单及指纹。Full、Continue和测量尾段分账，主能量为任务＋已启动推理收尾；同一任务对齐比较，同时呈现全部成功/超时。

## 提速方式

1. 每组常驻同一份模型/专家，目标两次加载完成80回合。
2. 40个MuJoCo环境先编译并验证XML/二进制重放，一组完成后，另一组复用相同清单/指纹的环境缓存；运行时仍重新创建环境和重置初态。
3. `--reuse-prefix-graphs`只复用相同精确前缀长度的CUDA图，仍最多17张；保留重叠长度，只构建缺失长度并淘汰不用的长度。每次请求继续刷新当前KV，attention仍按真实长度计算。这不是复用旧观测或旧KV。
4. 不启用跳过reset/校准的旧`--reuse-preparation`实验，不改去噪、控制周期、任务步数上限、成功判定或功耗模式。内存保护、GPU互斥、原始数据及失败退出保留。

新入口及缓存路由91项CPU检查通过。GPU固定输入验证`M4/20261104T010624.000054Z-prefix-cache-parity-2ec7919f`：同真实context/旧目标、三个固定噪声，原图与常驻图、重叠长度、淘汰和回切共12次对照，raw与VJP范数差均0；没有缺失长度时准备约0.70秒，新增一个长度约2.68–2.69秒（含三次回放核对），旧单任务17长度准备约35–40秒。是否能复用以及全量提速幅度以实际任务记录为准。模型固定、不改FlashRT。

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m4_acceptance.py \
  --candidate m3 --all-tasks --request-only-images --reuse-prefix-graphs \
  --low-full --three-frequency --idle-min-ms 50 --start-floor-ms 25 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh
```

结束后交付每任务成功/步数/耗时/总能量、Full与Continue分账、各集合成功率、共同成功任务与全部尝试的两种能量比较，以及相同初态和输入/首次动作检查。保留所有超时，不删除失败来提高成功率。

## 已启动

代码`83002b2`，后台服务`perceptbudget-all40-request-images-20261104.service`。会话目录`M4/20261104T010944.227352Z-paired-acceptance-3b809feb`，先准备40个环境并执行q30，再复用环境缓存执行动态组。主日志`/tmp/perceptbudget-all40-request-images-20261104.log`。当前处于运行阶段；正式结果以会话结束后的`comparison.json`、`comparison.csv`和逐回合原始记录为准。
