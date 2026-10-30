# Full/Fast 与 Continue 分账修正

本次实验使用 `--fast-advance`；保留旧标志作为兼容别名。FlashRT、模型和RTC算法未改。

1. Full/Fast全部高频计算。Full可直接高频启动或从低频恢复；Fast仅从低频Continue恢复后启动，且下一次刷新必须Full。
2. 原RTC提交边界减去Fast恢复至动作提交的预算及eta，向上取整。例如39−8=31；不再减Full预算，不重复减服务时间。窗外关键事件仍Full。
3. 选择器仅比较完整服务能量；Continue不并入Fast或Full。此策略不是全未来轨迹能量最优求解器，效果由闭环总能量判断。
4. 每回合新增service_continue_ledger.json：高频Full、恢复Full、恢复Fast，以及其余Continue。恢复至动作提交计服务，降频及提交后等待计Continue；未提交的末次请求计至ready。各段按相同VDD_IN读数积分，检查不重叠及总和一致。此为按时间活动划分的整板能量，包含并行控制与仿真，非GPU独占能量。

验证：75项相关测试通过，CLI确认Long6单回合/一次模型加载/V4。历史Long6完整功率记录离线分账为Continue80.453J、高频Full13.075J、恢复Full57.130J，合150.659J，与原整段积分一致；这只是账本验证，不是V4闭环结果。

新V4闭环待用户终端sudo执行；先测Long6，不并发启动其他GPU任务。
