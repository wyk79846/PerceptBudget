# H50 LIBERO 成品权重接入状态

用户选定：`lerobot/pi05_libero_finetuned_quantiles_v044`。
固定版本：`b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd`。

## 已确认

- 发布的训练与模型配置均为 `chunk_size=50`，不是将现有H10权重临时扩展到50。
- 包含8维state，state/action均QUANTILES归一化，附带独立预处理/反归一化统计文件。
- 配置还包含第三路empty camera、max_state_dim32、tokenizer_max_length200、N10。这些必须遵守配套处理逻辑，不直接套旧state80/N5配置。
- 新旧safetensors头各812项；新key去掉`model.`前缀后，与现有OpenPI格式键集合和张量shape完全一致。这仅说明结构可映射，不是数值或闭环兼容性验证。
- 权重7,473,096,344字节，发布的SHA256为`33d3e397a5623d862809cde2d544b4953cdc5550eb2e6d7c3b5556ae39cde48f`。

## 存放与下载

独立缓存：`/home/wyk/workspace/openpi_cache/openpi-assets/checkpoints/lerobot_pi05_libero_h50_quantiles_v044`。

下载入口：`scripts/fetch_libero_h50.py`。固定版本，可断点续传，逐文件大小/SHA256核对，状态写缓存中的`download_manifest.json`。后台服务`perceptbudget-fetch-libero-h50-proxy-20261020`，日志`/tmp/perceptbudget-fetch-libero-h50-20261020.log`。

首次服务没有代理、停在元数据阶段，已仅停止该服务并让替代进程继承现有终端代理；没有改变系统代理。目前8文件全部下载校验完成，manifest为`download_verified`，模型LFS SHA256一致。

## 已完成接入与单任务验证

- `perceptbudget/h50_checkpoint.py`提供独立profile与配套统计/state/image/action适配；suffix加载器仅映射`model.`键名，无第二份7GB转换权重，无新工作区，不改FlashRT。
- state正规化后补到32维，text容量200；图像使用float bilinear、不回舍入uint8；第三路相机原本mask=false，项目只保留有效双图；新模型反归一化不clip、不附加旧1e-6。
- 原生LeRobot state/prompt独立CPU核对通过，当前权重每个exact长度图/eager核对通过；不是原生完整LeRobot模型GPU逐层对齐或H50/N10完整发布设置复现。
- 四组同task0/H20/N5/β5/固定d5/seed7init0，q5与q8：新模型76步/10.053秒、74步/8.139秒；旧模型198步/23.098秒、210步/20.917秒，全成功，无断供/图回退。新模型单次推理约610–612ms，旧约561–565ms，任务提速来自动作效率而非模型计算更快。
- 四组初sim逐值同，2366独立记录检查全过。全套365 passed/10 skipped/178 subtests。
- 新模型首次在0控制步触发768MiB保护，该尝试标记invalid、排除。之后仅释放已空闲CPU分配器页重试成功，阈值不变；不声称全任务内存稳定。
- 结果：`/home/wyk/workspace/任务产出/perceptbudget/M1/20261020T074138.229989Z-rtc-checkpoint-comparison-7e9b8d6c/对照结果.md`。

## 后续建议（尚未执行）

1. 扩展新模型其他Spatial任务，不能从单任务推断成功率。
2. 需要更强复现结论时，补原生完整模型GPU同观测核对。
3. 真实异步已有单task0/q5验证（见下），后续需扩任务，不将受控d5或单任务成功冒充20Hz/总体成功率。

## 真实异步单任务补充

H20/N5/q5/β5、动态historymax估计、实际步数裁剪、不暂停仿真：task0 68步成功，5.721秒，推理中位631.23ms，实际14.25Hz，无断供/图回退/受控等待，241记录检查通过。实际后台延迟8～10步。记录为任务产出M1/20261020T075334.633718Z-rtc-handoff-comparison-6898873a/结论.md；仅单任务，不证明稳定20Hz或十任务质量。

旧权重和全局config保留，尚未切换默认模型，不训练，不宣称新模型的总体RTC成功率或唯一失败根因。
