# M3 comparison 内存中断续测

2026-10-28 正式比较在19/36回合后触发768 MiB可用内存保护。
已完成16次成功、3次Object task3任务超时；任务超时属于有效质量结果，全部保留。
模型 allocated 约6929 MiB、reserved约7044–7050 MiB，没有持续上升；
中断附近新增VS Code extensionHost及fileWatcher约850 MiB RSS，与外部内存竞争一致，尚未定位单次触发分配。
已完成回合的频率恢复与功率覆盖重新审查通过。没有修改FlashRT或降低保护线。

续测入口：

```bash
cd /home/wyk/workspace/PerceptBudget
/home/wyk/venvs/lutvla/bin/python -u scripts/run_m3.py --mode comparison \
  --resume-comparison /home/wyk/workspace/任务产出/perceptbudget/M3/20261028T014645.474204Z-comparison-0f04f20e
```

先关闭不用的远程编辑器窗口，减少后台内存竞争；不应直接杀掉全部node进程。
续测先审查旧计划、模型配置、逐回合系统审查、频率和功率覆盖；只编译剩余5任务的环境，
只加载一次模型补17回合。新目录引用原19回合，不修改原始结果。
整个比较总模型加载数为2；Object task3的D组跨运行，最终初始条件审查保留不一致标记，
不能把不一致的配对宣称为严格相同条件的节能差异。

## 独立复测一个已完成回合

`--resume-comparison <完成目录> --retest-comparison-cell libero_goal 6 C_independent`
只新增该回合，其他35条引用旧记录，`retested_original_episodes`保留原失败路径及结果。
新目录的汇总仅是复测视图；原始36回合正式比较不得被覆盖或当作没有发生该失败。
复测后必须比较首次预测与原A/B/D，确认之前的数值差异是否重现。
