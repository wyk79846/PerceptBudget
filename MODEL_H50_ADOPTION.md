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

首次服务没有代理、停在元数据阶段，已仅停止该服务并让替代进程继承现有终端代理；没有改变系统代理。小文件已核对，权重大文件下载进行中。检查manifest为`download_verified`才算下载完整；不要由存在incomplete文件宣称完成。

## 后续顺序

1. 完成大文件SHA256核对。
2. 先按配套LeRobot处理流程验证加载、观测和同步单任务，不借用旧归一化统计。
3. 项目侧权重名映射与输入适配，必要时保存转换来源与校验；不修改FlashRT上游，不新增工作区。
4. 同观测对照通过后，再做RTC固定延迟，最后才测真实异步。

旧权重和当前config保留，尚未切换默认模型、尚未运行新权重推理、不训练，不宣称新模型的RTC成功率。
