# QNN GPU Backend 构建与部署（Z-Image on Snapdragon 8 Elite）

> 本文档用于复现 **QNN GPU（Adreno 830）Transformer 推理路径**。
> 当前唯一数值正确的 Transformer GPU 路径是 **v10 全 FP32、10 个 3-layer 段**。
> HTP 仍使用 v6 FP16（待数值问题修复）；VAE GPU 路径尚未修复（FP16 输出 NaN），本文暂不涉及 VAE。

- 设备：OnePlus 13 / PJZ110 / SM8750 / Adreno 830 / HTP v79
- QAIRT：2.49.0.260730
- 最后验证日期：2026-08-18 night

---

## 1. 为什么是“v10 FP32 + 3-layer 分段”

1. **全 FP16 不可用**：QNN GPU 的 FP16 MatMul/FullyConnected 在 attention 投影处累加溢出为 inf/NaN；同一 FP16 ONNX 在 PC onnxruntime CPU 上输出有限，因此是 QNN FP16 累加实现问题。
2. **FP32 可消除溢出**：`--float_bitwidth 32` 后 frontend 无 NaN，与 ORT 参考 `corr=0.99999998`；全链 `noise_pred` 与 ORT 参考 `corr=0.999996`。
3. **必须 3-layer 分段**：6-layer FP32 段 `model.bin` 为 4.34GB，超过 aarch64 链接器 `R_AARCH64_ADR_PREL_PG_HI21` 4GB 相对寻址范围，`ld.lld` 报 out of range 失败。3-layer 段为 2.17GB，可正常链接。
4. HTP 后端对 FP32 图仍会走自己的 FP16 数值路径并产生 NaN；**v10 FP32 是 GPU 专用产物**。

---

## 2. 环境与路径

```bash
export QAIRT_PYTHON='E:\projects\zimage on phone\zimage-runtime\python312\python.exe'
export QAIRT_TMP_DIR='E:\projects\zimage on phone\zimage-runtime\qairt-tmp'
export PYTHONPATH='E:/qnn/lib/python'
export GRADLE_USER_HOME='E:\projects\zimage on phone\zimage-runtime\gradle-home'
export JAVA_HOME='C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot'
export MSYS_NO_PATHCONV=1
```

| 项 | 路径 |
|---|---|
| 项目根 | `E:\projects\zimage on phone\zimage on phone` |
| QAIRT junction | `E:\qnn`（脚本必须显式传 `-QnnSdkRoot E:\qnn`） |
| ONNX 导出源 | `build/transformer_segments_v10_maskfix_3layer/` |
| QNN 转换产物 | `build/qnn_transformer_v10_fp32/` |
| Android .so 产物 | `build/android/jniLibs-transformer-v10-fp32/arm64-v8a/` |
| APK | `android/app/build/outputs/apk/debug/app-debug.apk` |

---

## 3. 步骤 A：导出 3-layer ONNX 段

`tools/export_transformer_segments.py` 已包含 float-mask patch（避免 bool Cast/StridedSlice）。用 FP16 导出（QNN 转换时再 upcast 到 FP32）：

```bash
cd '/e/projects/zimage on phone/zimage on phone'
"$QAIRT_PYTHON" tools/export_transformer_segments.py \
  --model-root "tools/models/z-image-turbo/models/Tongyi-MAI--Z-Image-Turbo/snapshots/master" \
  --out-dir "build/transformer_segments_v10_maskfix_3layer" \
  --dtype fp16 \
  --layers-per-segment 3
```

产物：

```text
build/transformer_segments_v10_maskfix_3layer/
  frontend/model.onnx
  layers_00_02/model.onnx ... layers_27_29/model.onnx  (10 个)
  final/model.onnx
  manifest.json
```

注意：layer 段 ONNX 权重是 external data（目录里还有大量 `.weight/.bias` 文件），不能只拷贝 `model.onnx`。

---

## 4. 步骤 B：转换 QNN 全 FP32

一键脚本（已创建并验证）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\projects\zimage on phone\zimage on phone\tools\convert_transformer_v10_fp32_all.ps1" -QnnSdkRoot "E:\qnn"
```

等价的核心 converter 参数：

```bash
"$QAIRT_PYTHON" 'E:/qnn/bin/x86_64-windows-msvc/qnn-onnx-converter' \
  --input_network <segment>/model.onnx \
  --output_path <out>/model.cpp \
  --no_simplification \
  --float_bitwidth 32 \
  --float_bias_bitwidth 32 \
  --input_dim ...
```

各段 `--input_dim`：

| 段 | input_dim |
|---|---|
| frontend | `latent 1,16,64,64`、`timestep 1`、`cap_feats 512,2560`、`cap_mask 512` |
| layers_* | `unified_in 1,1536,3840`、`unified_freqs 1,1536,128`、`unified_mask 1,1536`、`adaln_input 1,256` |
| final | `unified_in 1,1536,3840`、`adaln_input 1,256` |

产物大小（应大致如此）：

```text
frontend/model.bin          2.92 GB
layers_*/model.bin          2.17 GB × 10
final/model.bin             5 MB
```

---

## 5. 步骤 C：构建 Android model `.so`

必须设置 `$env:QAIRT_PYTHON`，并显式 `-QnnSdkRoot E:\qnn`：

```bash
powershell -NoProfile -ExecutionPolicy Bypass -Command "\$env:QAIRT_PYTHON='E:\projects\zimage on phone\zimage-runtime\python312\python.exe'; & 'E:\projects\zimage on phone\zimage on phone\tools\build_transformer_qnn_libs.ps1' -QnnSdkRoot 'E:\qnn' -InputRoot 'E:\projects\zimage on phone\zimage on phone\build\qnn_transformer_v10_fp32' -OutputRoot 'E:\projects\zimage on phone\zimage on phone\build\android\jniLibs-transformer-v10-fp32\arm64-v8a' -Segments @('frontend','layers_00_02','layers_03_05','layers_06_08','layers_09_11','layers_12_14','layers_15_17','layers_18_20','layers_21_23','layers_24_26','layers_27_29','final') -ForceRebuild"
```

构建脚本已经处理：
- `model.bin` 的 tar 解包（Windows tar 不能用，脚本用 Python tarfile）
- objcopy 必须相对路径 `obj/binary/<file>.raw`
- 复制 `linux/QnnModelPal.cpp`（提供 `qnn_wrapper_api::strnDup`）

产物：

```text
build/android/jniLibs-transformer-v10-fp32/arm64-v8a/
  libqnn_transformer_frontend.so      2.92 GB
  libqnn_transformer_layers_*.so      2.17 GB × 10
  libqnn_transformer_final.so         5 MB
```

总大小约 23GB。

---

## 6. 步骤 D：构建/确认 APK 的 GPU 支持

```bash
cd '/e/projects/zimage on phone/zimage on phone/android'
'/c/Users/Max/.gradle/wrapper/dists/gradle-8.11.1-bin/bpt9gzteqjrbo1mjrsomdt32c/gradle-8.11.1/bin/gradle.bat' assembleDebug
```

构建前确认以下代码存在：

- `AndroidManifest.xml`：
  ```xml
  <uses-native-library android:name="libOpenCL.so" android:required="false" />
  ```
- `app/build.gradle`：
  ```groovy
  packaging { jniLibs { excludes += "lib/**/libOpenCL*.so" } }
  ```
- `build/android/apk-jniLibs/arm64-v8a/` 中不得有 `libOpenCL.so` / `libOpenCL_adreno.so`。
- `zimage_runtime.cpp`：`backendKind=2` 走 `libQnnGpu.so`；`nativeTransformerProbe` 检测 frontend dtype：FP32 → v10 10×3-layer，FP16 → v6 5×6-layer。

---

## 7. 步骤 E：部署到设备

推荐用部署脚本（把 QNN SDK 库、v10 Transformer、text FP16、VAE、assets 全部推到 external runtime）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\projects\zimage on phone\zimage on phone\tools\deploy_android_runtime.ps1" -Adb adb -TransformerLibRoot "E:\projects\zimage on phone\zimage on phone\build\android\jniLibs-transformer-v10-fp32" -TextEncoderLibRoot "E:\projects\zimage on phone\zimage on phone\build\android\jniLibs-text-fp16"
```

手工增量更新 v10 库（快，适合已经部署过 runtime 的设备）：

```bash
export MSYS_NO_PATHCONV=1
REMOTE=/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/lib/arm64-v8a
adb shell mkdir -p "$REMOTE"
for f in build/android/jniLibs-transformer-v10-fp32/arm64-v8a/libqnn_transformer_*.so; do
  adb push "$f" "$REMOTE/$(basename "$f")"
done
adb shell chmod -R a+rX /storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime
```

设置 GPU 后端开关：

```bash
adb shell mkdir -p /storage/emulated/0/Android/data/com.example.zimage/files/probe
adb shell "echo gpu > /storage/emulated/0/Android/data/com.example.zimage/files/probe/backend.txt"
```

安装并启动：

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.example.zimage
adb shell am start -n com.example.zimage/.MainActivity
```

App 首次启动会把 external 下的新 `.so` 按大小比较复制到私有 `filesDir/zimage-runtime`，v10 约 23GB，需要等待几分钟。

---

## 8. 步骤 F：验证

### 8.1 App 日志关键行

```bash
adb shell 'run-as com.example.zimage grep -nE "zimage runtime init|OpenCL probe|GPU backendCreate|GPU contextCreate|transformer frontend dtype" files/zimage-runtime/jni.log'
```

预期：

```text
=== zimage runtime init === backend=gpu
GPU backend lib: ok
OpenCL probe: dlopen libOpenCL.so ok
OpenCL probe: clGetPlatformIDs rc=0 num=1
GPU backendCreate=0
GPU deviceCreate=0 dev=set
GPU contextCreate=0
transformer frontend dtype=fp32
```

### 8.2 全链 probe 预期

`progress.txt` 最终 `transformer: 全部完成 12/12`；`jni.log` 末尾应有：

```text
transformer probe: frontend ms=818 unified=fp32[...] min=-61.46 max=739.95 mean=0.0412 ...
layers_00_02 ms=1834 out=fp32[...] min=-61.42 max=1023.7 mean=0.0731
...
layers_27_29 ms=1897 out=fp32[...] min=-116.6 max=5370 mean=0.3059
final ms=21 noise=fp32[65536] min=-6.66 max=6.72 mean=-0.0603
```

已验证结果（2026-08-18 night）：

| 对比 | 结果 |
|---|---|
| app `probe_noise_pred.raw` vs CLI qnn-net-run | maxdiff = 0 |
| app `probe_noise_pred.raw` vs ORT 参考 | corr = 0.999996、maxabs = 0.019 |

### 8.3 独立 CLI 验证（可选）

先验证 frontend：

```powershell
powershell -File tools\test_qnn_gpu_frontend.ps1 -Backend gpu
```

该脚本自动推 `qnn-net-run`、`libQnnGpu.so`、v10 frontend 和 probe 输入，并执行。

全链 CLI 接续参考：
- frontend 输出 `unified.raw[1,1536,3840]` → transpose 为 `[1,3840,1536]` 作为下一层 `unified_in`；
- `unified_freqs` 用 `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw`；
- `unified_mask`、`adaln_input` 直接用 frontend 输出；
- 10 个 layer 段依次执行，`unified_out` 直接作为下一段 `unified_in`；
- final 输入 `unified_in + adaln_input`，输出 `noise_pred.raw`。

---

## 9. 已知问题与边界

1. **GPU VAE 尚未修复**：FP16 VAE 在 GPU 上输出全 NaN（HTP 正常）。本 GPU 路径当前只覆盖 Transformer。
2. **HTP 不要用 v10 FP32**：HTP 对 FP32 图仍产生 NaN；HTP 保留 v6 FP16（同样有已知数值风险）。
3. **v10 只支持 3-layer 段**：6-layer FP32 段会触发 aarch64 4GB 链接重定位错误。
4. **每次全链前各段都要 prepare/finalize**：v10 各段 execute 约 1.7–2s，但每段 prepare/finalize 另计；后续可用 context binary 缓存优化。
5. **OpenCL 打包红线**：任何情况下不要重新把 `/vendor/lib64/libOpenCL*.so` 放进 APK。
6. **不要用 `qnn-onnx-converter --backend`**：该工具没有这个参数。
7. 多输入 qnn-net-run 的 input_list 必须一行 `name:=file` 空格分隔，不是每行一个文件。

---

## 10. 相关文件

- 转换脚本：`tools/convert_transformer_v10_fp32_all.ps1`
- 构建脚本：`tools/build_transformer_qnn_libs.ps1`
- 前端测试脚本：`tools/test_qnn_gpu_frontend.ps1`
- 部署脚本：`tools/deploy_android_runtime.ps1`
- 主交接文档：`PROJECT_HANDOFF.md`
- 完整历史：`PROJECT_HANDOFF_ARCHIVE.md`
