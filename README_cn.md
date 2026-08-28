# Z-Image on Snapdragon 8 Elite

把阿里 **Tongyi-MAI/Z-Image-Turbo** 文生图模型完整跑在 **OnePlus 13 (SM8750 / Snapdragon 8 Elite)** 手机上：
CPU tokenizer → Qwen3 文本编码器 (QNN HTP FP16) → Z-Image Transformer DiT (QNN GPU FP32) → FlowMatch Euler 调度器 (CPU) → VAE 解码 (QNN HTP FP16)，全部在 **Android app 内**（无云端）。

已实现：**端到端出图**（512×512，8 步，首帧 ~580s），Material 3 UI，生成历史持久化，采样参数设置。

> **测试环境声明**：本项目仅在 **OnePlus 13 · ColorOS 16.0.9**（SM8750 / Snapdragon 8 Elite）上验证通过。其他设备/系统版本未经测试，QNN 后端行为、LMK 内存上限、驱动兼容性可能不同，无法保证运行结果。

> **维护声明**：本项目全部代码由 AI 辅助（vibecoding）产出，**无后续维护计划**。仅供学习参考，**谨慎使用**——不保证正确性、安全性，也不承诺修复问题。**如有需要，欢迎 fork 本项目。**

**[English README →](README.md)**

> 模型权重与 Qualcomm QNN SDK 二进制**不随仓库分发**（体积大且含专有许可），首次本地运行需按本文档准备。以下命令中的路径为通用占位，请替换为你自己的环境路径。

---

## 目录

1. [架构总览](#1-架构总览)
2. [前置准备：硬件与软件](#2-前置准备硬件与软件)
3. [三步跑通：构建 → 部署 → 生成](#3-三步跑通构建--部署--生成)
4. [模型产物准备（首次一次性）](#4-模型产物准备首次一次性)
5. [设备端 Runtime 部署](#5-设备端-runtime-部署)
6. [App 使用说明](#6-app-使用说明)
7. [性能与调优](#7-性能与调优)
8. [常见问题 FAQ](#8-常见问题-faq)
9. [项目结构](#9-项目结构)
10. [参考文档](#10-参考文档)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        Android App (OnePlus 13)             │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐    │
│  │  MainActivity │──▶│  JNI  zimage_runtime.cpp         │    │
│  │  (Kotlin M3)  │   │  ┌────────────────────────────┐  │    │
│  └──────────────┘   │  │ CPU: Tokenizer (BPE)        │  │    │
│                     │  │ CPU: Scheduler (Euler 8步)  │  │    │
│                     │  │ QNN HTP: Qwen3 4.0B FP16    │  │    │
│                     │  │ QNN GPU: Z-Image 6.15B FP32 │  │    │
│                     │  │ QNN HTP: VAE Decoder 84M    │  │    │
│                     │  └────────────────────────────┘  │    │
│                     └───────────────┬──────────────────┘    │
└─────────────────────────────────────┼───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │  QNN SDK (设备端)      │  模型产物 (设备端)      │
              │  libQnnHtp/Gpu.so     │  12 段 transformer     │
              │  + skel/hexagon       │  ctxbin  (~25GB)       │
              └───────────────────────┴───────────────────────┘
```

| 组件 | 参数量 | 后端/精度 | 状态 |
|---|---|---|---|
| Tokenizer (Qwen BPE + chat template) | - | C++ CPU（APK assets 内嵌词表） | ✅ 与 HF 输出一致 |
| Qwen3 文本编码器 | 4.0B | QNN **HTP FP16**（7 段） | ✅ corr=0.977 |
| Z-Image Transformer (DiT) | 6.15B | QNN **GPU FP32**（12 段） | ✅ corr=0.999996 |
| Scheduler (FlowMatchEuler) | - | C++ CPU（shift=3, 8 步） | ✅ |
| VAE Decoder | 84M | QNN **HTP FP16** | ✅ |

**为什么 Transformer 用 GPU FP32？** 全 FP16 在 QNN 的 MatMul/attention 投影累加溢出成 inf/NaN（PC ORT 同模型有限）。FP32 完全消除（corr=0.999996）。HTP 对 FP32 图仍走内部 FP16 路径也 NaN，所以 transformer 是 **GPU 专用产物**。

**为什么 12 段？** 6-layer FP32 段 model.bin 达 4.34GB，超过 aarch64 链接器相对寻址 4GB 上限，3-layer 段 2.17GB 可链接。

---

## 2. 前置准备：硬件与软件

### 硬件

| 项 | 要求 |
|---|---|
| 手机 | OnePlus 13（ColorOS 16.0.9，**本项目唯一验证环境**）/ 理论上 SM8750 设备可用（Snapdragon 8 Elite，Adreno 830 GPU + Hexagon HTP v79） |
| 手机存储 | ≥ 60GB 空闲（模型产物 ~25GB ctxbin + ~40GB .so） |
| PC | Windows（本流程验证于 Windows 11 + Git Bash），USB 调试开启 |

### 软件

| 项 | 说明 |
|---|---|
| Android SDK + NDK + CMake | 用于构建 APK |
| JDK 17 | 构建 Gradle 用（Microsoft OpenJDK 17 已验证） |
| Gradle 8.11.1 | 通过项目 wrapper 使用（`./gradlew`） |
| adb | Android 平台工具 |
| QNN / QAIRT SDK | 2.49.0（Qualcomm AI Engine Direct） |
| Python 3.10+ | 模型导出/转换脚本用（torch/diffusers/transformers/onnxruntime） |

> 路径全部使用通用占位符，请根据你的环境替换。

---

## 3. 三步跑通：构建 → 部署 → 生成

### 3.1 构建 APK

```bash
cd android
# 用本机 Gradle 8.11.1 构建（增量 ~5s）
gradle assembleDebug
```

> 构建前请自行设置好 JDK 17、Gradle 8.11.1、adb 的环境变量。

产物：`android/app/build/outputs/apk/debug/app-debug.apk`

### 3.2 安装启动

```bash
export MSYS_NO_PATHCONV=1   # git-bash 必需，否则 /data 路径被转换
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.example.zimage
adb shell am start -n com.example.zimage/.MainActivity
```

### 3.3 生成一张图

1. 打开 app，等「正在加载运行时…」结束（首次冷启恢复 ctxbin 约 1-2 分钟）
2. 输入提示词（支持中英文，如 `a red paper lantern` / `海滩边玩水的少女`）
3. 点「生成」→ 环心计时开始，底部进度条显示当前阶段
4. 完成后图片显示在画布，自动存入历史栏（app 私有目录，重启不丢）
5. 长按历史缩略图 → 保存到相册 / 分享 / 删除

---

## 4. 模型产物准备（首次一次性）

模型权重与 QNN SDK **不随仓库分发**。首次运行需完成转换（参考 `docs/gpu-backend-build.md` 与 `docs/backend-plan.md` 全流程），得到以下产物：

### 4.1 模型来源

- 模型：ModelScope **Tongyi-MAI/Z-Image-Turbo**（`tools/download_model.py` 用 ModelScope 拉取，配置见 `config/modelscope-z-image-turbo.json`）
- 参考实现：diffusers `ZImagePipeline`（scheduler 配置 shift=3.0、VAE scaling_factor=0.3611 等来自模型仓 config.json）

### 4.2 转换产物清单（build/ 下）

| 目录 | 内容 | 用途 |
|---|---|---|
| `build/android/jniLibs-transformer-v10-fp32/arm64-v8a/` | v10 FP32 全 12 段 .so（~23GB） | transformer（GPU） |
| `build/android/jniLibs-text-fp16/arm64-v8a/` | text embedding + 6 组 .so | 文本编码器（HTP） |
| `build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so` | VAE 修正版 | VAE（HTP） |
| `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw` | RoPE 频率表 | transformer 输入 |
| `build/text_encoder_precomp/` | cos/sin/causal_mask raw | text 输入 |
| `android/app/src/main/assets/qwen_vocab.tsv` + `qwen_merges.txt` | tokenizer 词表（已入库） | tokenizer |

> 转换工具链全部在 `tools/`（export_*.py / convert_*.ps1 / generate_*_calibration.py），转换注意 `--preserve_io datatype` 与 3-layer 分段（见 docs）。

### 4.3 预编译 ctxbin（关键优化，可选但强烈建议）

app 内编译 12 段 transformer 会触发 ColorOS LMK（内存墙）。**正解：在设备上用 shell 进程预编译**（不受 app LMK 限制）：

```bash
# 设备端（/data/local/tmp 下，shell 权限）
adb shell 'cd /data/local/tmp/gputest && ./qnn-context-binary-generator \
  --model frontend.so --backend libQnnGpu.so \
  --binary_file frontend.ctxbin --output_dir ./ctxbins'
# 产物 frontend.ctxbin.bin（注意自动加 .bin 后缀），改名后部署
```

全部 12 段生成后放到设备外部目录：

```bash
adb shell 'cp /data/local/tmp/ctxbins/*.ctxbin \
  /storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/segbin/'
```

> ⚠️ 工具只接受**相对路径**（须 cd 进输出目录）；输出自动带 `.bin` 后缀需改名。完整过程见 `PROJECT_HANDOFF.md` §9。

---

## 5. 设备端 Runtime 部署

模型 .so / assets 通过 `tools/deploy_android_runtime.ps1` 推到设备外部目录（app 首启按"大小不同才复制"拷入私有目录）：

```bash
cd <项目根目录>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/deploy_android_runtime.ps1
# GPU/FP32 时覆盖（默认是 FP16 v6）：
powershell.exe -File tools/deploy_android_runtime.ps1 `
  -TransformerLibRoot build\android\jniLibs-transformer-v10-fp32 `
  -TextEncoderLibRoot build\android\jniLibs-text-fp16
```

切换后端（gpu=当前生产配置）：

```bash
adb shell "printf 'gpu' > /storage/emulated/0/Android/data/com.example.zimage/files/probe/backend.txt"
```

查看运行日志 / 进度：

```bash
adb shell 'run-as com.example.zimage cat files/zimage-runtime/jni.log'
adb shell 'run-as com.example.zimage cat files/zimage-runtime/progress.txt'
```

---

## 6. App 使用说明

Material 3 UI，状态机 `INITIALIZING → IDLE → GENERATING → DONE` 驱动：

| 界面元素 | 功能 |
|---|---|
| 提示词输入框 | 多行，支持自动换行 |
| 「生成」按钮 | 开始生成（GENERATING 时禁用） |
| 画布 | 三态：空态 / 进度环（环心计时）/ 结果图 |
| 进度环 | 环心显示生成耗时 mm:ss |
| 历史栏 | 每次生成自动保存（时间戳命名 PNG），重启不丢 |
| 历史长按 | 保存到相册 / 分享 / 删除 |
| 设置（齿轮） | 采样步数 1-16 / 随机种子开关 / 固定种子值 |
| 诊断（ℹ️） | 复制全部诊断信息 / 导出日志到 Download/Z-Image/ |
| 图片点击 | 全屏预览 |

---

## 7. 性能与调优

| 指标 | 当前值 | 目标 |
|---|---|---|
| 首帧（冷启） | ~580s（首张 583.7s，次张 522.6s） | <60s |
| transformer 占比 | 558.9s / 583.7s（96%） | - |
| 内存峰值 | RSS 10GB → 200MB（rebuild 后回落） | <8GB 安全线 |

**已实现的优化**：
- 每 4 段 release 后销毁重建 GPU backend（kgsl dma-buf 缓存挂在 backend 生命周期上，RSS 瞬间回落）
- 12 段 ctxbin 预编译（app 内无编译峰值）

**下一步优化方向**：
1. Kernel repo 磁盘缓存（`QNN_GPU_CONTEXT_CONFIG_OPTION_KERNEL_REPO_DIR`），backend 重建后 kernel 秒载
2. rebuild 间隔 4→5 段（峰值 ~12GB 边缘）
3. 首帧 580s → 目标 3-5 分钟

---

## 8. 常见问题 FAQ

**Q: 生成时 app 被杀（无闪退界面，进程消失）**
A: ColorOS LMK 杀进程。前台 app 内存上限约 8-9GB。确认已用预编译 ctxbin（§4.3），且 12 段按 4 段间隔 rebuild。

**Q: 报 6001 INVALID_HANDLE**
A: 图操作句柄必须来自图所属的 `QnnSet`（双 backend 下混用全局句柄会这样）。另外 `GraphInfo` 字段顺序必须与 SDK 的 `QnnWrapperUtils.hpp` 一致（`{graph, graphName, inputTensors, ...}`）。

**Q: 输出全垃圾/NaN**
A: 检查 I/O dtype 转换是否加了 `--preserve_io datatype`；Transformer 必须 FP32（FP16 在 QNN 累加溢出）；VAE 直接喂原始 latent（shiftfix 图内已含归一化，勿二次处理）。

**Q: `libQnnHtp.so` transport 14001**
A: `ADSP_LIBRARY_PATH` 必须指向含 `libQnnHtpV79Skel.so` 的目录（设备 CLI 验证时）。

**Q: 第三方安装器重装后 app 无法生成**
A: 重装会清空外部数据目录。恢复：116 个 libs（含 hexagon-v79 skel）+ 8 assets + backend.txt=gpu + 12 个 ctxbin。`libQnnSystem.so` 必须用 aarch64-android 完整版（4,072,160 字节），裁剪版（217KB）会导致 binaryInfo 全失败。

**Q: adb 命令报 `no such file or directory`**
A: git-bash 需 `export MSYS_NO_PATHCONV=1`，否则 `/data/...` 路径被 MSYS 转换。

**Q: patch 后构建失败**
A: patch 工具会把行尾转 CRLF，`tr -d '\r' < f > /tmp/x && mv /tmp/x f` 归一化（C++ 文件必须 LF）。

---

## 9. 项目结构

```
├── android/                      # Android Studio 工程
│   ├── app/src/main/
│   │   ├── java/com/example/zimage/
│   │   │   ├── MainActivity.kt   # M3 UI + 状态机 + 生成调度
│   │   │   ├── HistoryStore.kt   # 持久化历史（时间戳 PNG + JSON 索引）
│   │   │   ├── SettingsStore.kt  # 采样参数持久化
│   │   │   ├── UiUtil.kt         # 保存/分享/预览/剪贴板/日志导出
│   │   │   └── ZImageRuntime.kt  # JNI 封装
│   │   ├── cpp/
│   │   │   ├── zimage_runtime.cpp # ★ 核心：tokenizer→text→DiT→scheduler→VAE
│   │   │   ├── CMakeLists.txt
│   │   │   └── vndksupport.*      # QNN 库链接支持
│   │   ├── assets/               # tokenizer 词表 + GPU probe（已入库）
│   │   └── res/                  # M3 主题/布局/图标/文案
│   ├── build.gradle / settings.gradle / gradle.properties
│   └── local.properties          # 本机 SDK 路径（不入库）
├── tools/                        # 模型导出/量化/转换/部署脚本（46 个）
├── config/                       # ModelScope 模型清单
├── docs/                         # bring-up 日志、backend 构建、IO 契约
├── PROJECT_HANDOFF.md            # ★ 完整交接文档（含全部踩坑记录）
└── README.md
```

---

## 10. 参考文档

- `PROJECT_HANDOFF.md` — 完整交接文档（架构细节、数值契约、全部踩坑与解法）
- `docs/gpu-backend-build.md` — GPU backend v10 转换到设备验证全套命令
- `docs/backend-plan.md` — 组件方案
- `docs/qnn-v3-io-contract.md` — QNN I/O 布局契约
- diffusers 参考：`pipeline_z_image.py` + `scheduling_flow_match_euler_discrete.py`

---

## License / 免责

**项目自身代码**（Kotlin 源码、`zimage_runtime.cpp`、资源、脚本、文档）采用
**Apache License 2.0** 授权 — 见 [`LICENSE`](LICENSE)。Copyright © 2026 imx123。

> 全部代码由 vibecoding（AI 辅助）产出，**无后续维护**，按现状提供，自行承担风险（见文首声明）。

**不适用本许可的部分**：
- 模型权重（Z-Image-Turbo）— 归 ModelScope/Tongyi-MAI 许可管理。
- Qualcomm QNN SDK 二进制（APK 内 `libQnnHtpV79Stub.so`）— Qualcomm 专有；
  仅以目标代码形式、作为 app 一部分分发，遵循 Qualcomm AI Stack License。见
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
- 内置第三方组件（LiteRT、Material 3、AndroidX、Qwen tokenizer 数据）— Apache 2.0，
  见 `THIRD_PARTY_NOTICES.md`。
