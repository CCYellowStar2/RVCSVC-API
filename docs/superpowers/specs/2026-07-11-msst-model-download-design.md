# MSST 模型按需下载

## 目标

让未随代码仓库分发的 `MSST BS-Roformer` 权重，在用户首次选择该分离模式时从项目的 ModelScope 仓库自动获取。

## 行为

- 应用启动时继续仅检查并下载默认的 VR-HP5 权重。
- 用户选择 `MSST BS-Roformer` 后，后端先检查 `msst/pretrain/vocal_models/bs_roformer_karaoke_frazer_becruily.ckpt`。
- 权重缺失时，使用现有的下载重试方式从以下地址下载到该路径：
  `https://modelscope.cn/models/CCYellowStar/bs_roformer_karaoke_frazer_becruily/resolve/master/bs_roformer_karaoke_frazer_becruily.ckpt`。
- 下载成功后在同一次请求中继续执行 MSST 分离；下载失败则抛出包含模型路径和下载原因的异常。
- 配置 YAML 仍随代码仓库提供，不参与下载。

## 位置与边界

下载判断放在 `msst_backend.py` 的 MSST 推理入口，使 RVC 与 SVC 共用同一份逻辑，避免两个应用入口产生重复代码。下载函数使用标准库和 HTTP 流式写入，创建目标目录，并以临时文件完成下载后再原子替换，避免中断后留下被误认为完整的权重文件。

## 测试

为下载辅助函数增加单元测试，覆盖：本地已有有效文件时跳过请求；下载成功时写入目标路径；下载失败时不会留下目标模型文件。现有 MSST 路径解析与分离调用测试继续执行。
