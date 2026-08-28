# Z-Image on Snapdragon 8 Elite

Run Alibaba's **Tongyi-MAI/Z-Image-Turbo** text-to-image model fully on-device on a **OnePlus 13 (SM8750 / Snapdragon 8 Elite)**:
CPU tokenizer → Qwen3 text encoder (QNN HTP FP16) → Z-Image Transformer DiT (QNN GPU FP32) → FlowMatch Euler scheduler (CPU) → VAE decoder (QNN HTP FP16) — all inside a single **Android app** (no cloud).

Implemented: **end-to-end image generation** (512×512, 8 steps, first frame ~580s), Material 3 UI, persistent generation history, sampling parameter settings.

> **Tested-environment disclaimer**: This project has only been verified on **OnePlus 13 · ColorOS 16.0.9** (SM8750 / Snapdragon 8 Elite). Other devices/OS versions are untested; QNN backend behavior, LMK memory limits, and driver compatibility may differ, and results cannot be guaranteed.

> Model weights and Qualcomm QNN SDK binaries are **not distributed with this repository** (large size + proprietary licenses). You must prepare them locally following this guide. All paths below are generic placeholders — replace with your own environment.

**[中文版 README →](README_cn.md)**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites: Hardware & Software](#2-prerequisites-hardware--software)
3. [Three Steps to Run: Build → Deploy → Generate](#3-three-steps-to-run-build--deploy--generate)
4. [Preparing Model Artifacts (one-time)](#4-preparing-model-artifacts-one-time)
5. [On-Device Runtime Deployment](#5-on-device-runtime-deployment)
6. [App Usage](#6-app-usage)
7. [Performance & Tuning](#7-performance--tuning)
8. [FAQ](#8-faq)
9. [Project Structure](#9-project-structure)
10. [References](#10-references)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Android App (OnePlus 13)             │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐    │
│  │  MainActivity │──▶│  JNI  zimage_runtime.cpp         │    │
│  │  (Kotlin M3)  │   │  ┌────────────────────────────┐  │    │
│  └──────────────┘   │  │ CPU: Tokenizer (BPE)        │  │    │
│                     │  │ CPU: Scheduler (Euler 8-step)│  │    │
│                     │  │ QNN HTP: Qwen3 4.0B FP16    │  │    │
│                     │  │ QNN GPU: Z-Image 6.15B FP32 │  │    │
│                     │  │ QNN HTP: VAE Decoder 84M    │  │    │
│                     │  └────────────────────────────┘  │    │
│                     └───────────────┬──────────────────┘    │
└─────────────────────────────────────┼───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │  QNN SDK (on-device)  │  Model artifacts      │
              │  libQnnHtp/Gpu.so     │  12 transformer       │
              │  + skel/hexagon       │  ctxbin (~25GB)       │
              └───────────────────────┴───────────────────────┘
```

| Component | Params | Backend / Precision | Status |
|---|---|---|---|
| Tokenizer (Qwen BPE + chat template) | - | C++ CPU (vocab bundled in APK assets) | ✅ matches HF output |
| Qwen3 text encoder | 4.0B | QNN **HTP FP16** (7 segments) | ✅ corr=0.977 |
| Z-Image Transformer (DiT) | 6.15B | QNN **GPU FP32** (12 segments) | ✅ corr=0.999996 |
| Scheduler (FlowMatchEuler) | - | C++ CPU (shift=3, 8 steps) | ✅ |
| VAE Decoder | 84M | QNN **HTP FP16** | ✅ |

**Why GPU FP32 for the transformer?** Full FP16 overflows to inf/NaN in QNN's MatMul/attention projection accumulation (the same model is finite on PC ORT). FP32 eliminates this entirely (corr=0.999996). HTP still routes FP32 graphs through an internal FP16 path and NaNs, so the transformer is a **GPU-only artifact**.

**Why 12 segments?** A 6-layer FP32 segment's model.bin reaches 4.34GB, exceeding the aarch64 linker's 4GB relative-addressing limit; 3-layer segments at 2.17GB link fine.

---

## 2. Prerequisites: Hardware & Software

### Hardware

| Item | Requirement |
|---|---|
| Phone | OnePlus 13 (ColorOS 16.0.9, **the only verified environment**) / theoretically any SM8750 device (Snapdragon 8 Elite: Adreno 830 GPU + Hexagon HTP v79) |
| Phone storage | ≥ 60GB free (~25GB ctxbin + ~40GB .so model artifacts) |
| PC | Windows (validated on Windows 11 + Git Bash), USB debugging enabled |

### Software

| Item | Notes |
|---|---|
| Android SDK + NDK + CMake | To build the APK |
| JDK 17 | For Gradle build (Microsoft OpenJDK 17 verified) |
| Gradle 8.11.1 | Via your local Gradle install (the project has no wrapper) |
| adb | Android platform tools |
| QNN / QAIRT SDK | 2.49.0 (Qualcomm AI Engine Direct) |
| Python 3.10+ | For model export/conversion scripts (torch/diffusers/transformers/onnxruntime) |

> All paths are generic placeholders — replace with your environment.

---

## 3. Three Steps to Run: Build → Deploy → Generate

### 3.1 Build the APK

```bash
cd android
# Build with your local Gradle 8.11.1 (incremental ~5s)
gradle assembleDebug
```

> Make sure JDK 17, Gradle 8.11.1, and adb are set up on your PATH before building.

Output: `android/app/build/outputs/apk/debug/app-debug.apk`

### 3.2 Install & Launch

```bash
export MSYS_NO_PATHCONV=1   # required in git-bash, otherwise /data paths get mangled
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.example.zimage
adb shell am start -n com.example.zimage/.MainActivity
```

### 3.3 Generate an Image

1. Open the app and wait for "loading runtime…" to finish (first cold start restores ctxbin, ~1-2 min)
2. Enter a prompt (Chinese and English supported, e.g. `a red paper lantern` / `海滩边玩水的少女`)
3. Tap **Generate** — timer starts in the ring center, bottom bar shows the current stage
4. When done, the image appears on the canvas and is auto-saved to history (app-private dir, survives restart)
5. Long-press a history thumbnail → Save to gallery / Share / Delete

---

## 4. Preparing Model Artifacts (one-time)

Model weights and the QNN SDK are **not distributed with this repo**. You must convert them once before first run (see `docs/gpu-backend-build.md` and `docs/backend-plan.md` for the full workflow).

### 4.1 Model Source

- Model: ModelScope **Tongyi-MAI/Z-Image-Turbo** (fetched via `tools/download_model.py`; config in `config/modelscope-z-image-turbo.json`)
- Reference implementation: diffusers `ZImagePipeline` (scheduler shift=3.0, VAE scaling_factor=0.3611, etc. from the model repo's config.json)

### 4.2 Conversion Artifacts (under build/)

| Directory | Content | Purpose |
|---|---|---|
| `build/android/jniLibs-transformer-v10-fp32/arm64-v8a/` | v10 FP32 all 12 segments .so (~23GB) | transformer (GPU) |
| `build/android/jniLibs-text-fp16/arm64-v8a/` | text embedding + 6 groups .so | text encoder (HTP) |
| `build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so` | VAE fixed version | VAE (HTP) |
| `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw` | RoPE frequency table | transformer input |
| `build/text_encoder_precomp/` | cos/sin/causal_mask raw | text inputs |
| `android/app/src/main/assets/qwen_vocab.tsv` + `qwen_merges.txt` | tokenizer vocab (already in repo) | tokenizer |

> Toolchain lives in `tools/` (export_*.py / convert_*.ps1 / generate_*_calibration.py). Remember `--preserve_io datatype` and 3-layer segmentation (see docs).

### 4.3 Precompile ctxbin (key optimization, optional but strongly recommended)

Compiling the 12 transformer segments inside the app triggers ColorOS LMK (memory wall). **The fix: precompile on-device in a shell process** (not subject to the app's LMK limit):

```bash
# On device (under /data/local/tmp, shell permission)
adb shell 'cd /data/local/tmp/gputest && ./qnn-context-binary-generator \
  --model frontend.so --backend libQnnGpu.so \
  --binary_file frontend.ctxbin --output_dir ./ctxbins'
# Output is frontend.ctxbin.bin (note the auto-appended .bin suffix); rename, then deploy
```

After generating all 12 segments, place them in the device's external dir:

```bash
adb shell 'cp /data/local/tmp/ctxbins/*.ctxbin \
  /storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/segbin/'
```

> ⚠️ The tool only accepts **relative paths** (cd into the output dir first); outputs get an automatic `.bin` suffix that must be renamed. Full walkthrough in `PROJECT_HANDOFF.md` §9.

---

## 5. On-Device Runtime Deployment

Model .so / assets are pushed to the device external dir via `tools/deploy_android_runtime.ps1` (the app copies them into its private dir on first launch, only when sizes differ):

```bash
cd <project-root>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/deploy_android_runtime.ps1
# Override for GPU/FP32 (default is FP16 v6):
powershell.exe -File tools/deploy_android_runtime.ps1 `
  -TransformerLibRoot build\android\jniLibs-transformer-v10-fp32 `
  -TextEncoderLibRoot build\android\jniLibs-text-fp16
```

Switch backend (gpu = current production config):

```bash
adb shell "printf 'gpu' > /storage/emulated/0/Android/data/com.example.zimage/files/probe/backend.txt"
```

View runtime log / progress:

```bash
adb shell 'run-as com.example.zimage cat files/zimage-runtime/jni.log'
adb shell 'run-as com.example.zimage cat files/zimage-runtime/progress.txt'
```

---

## 6. App Usage

Material 3 UI driven by the state machine `INITIALIZING → IDLE → GENERATING → DONE`:

| UI element | Function |
|---|---|
| Prompt input | Multi-line, wraps automatically |
| Generate button | Starts generation (disabled while GENERATING) |
| Canvas | Three states: empty / progress ring (elapsed mm:ss in center) / result image |
| Progress ring | Shows elapsed generation time in the center |
| History bar | Every generation auto-saved (timestamped PNG), survives restart |
| History long-press | Save to gallery / Share / Delete |
| Settings (gear) | Sampling steps 1-16 / random seed toggle / fixed seed value |
| Diagnostics (ℹ️) | Copy all diagnostics / export log to Download/Z-Image/ |
| Tap image | Fullscreen preview |

---

## 7. Performance & Tuning

| Metric | Current | Target |
|---|---|---|
| First frame (cold start) | ~580s (583.7s first image, 522.6s second) | <60s |
| Transformer share | 558.9s / 583.7s (96%) | - |
| Peak memory | RSS 10GB → 200MB (after rebuild) | <8GB safety line |

**Optimizations already in place**:
- Destroy + rebuild the GPU backend every 4 released segments (kgsl dma-buf cache hangs on the backend lifecycle; RSS drops instantly)
- All 12 ctxbin segments precompiled (no compile peak inside the app)

**Next optimization directions**:
1. Kernel repo disk cache (`QNN_GPU_CONTEXT_CONFIG_OPTION_KERNEL_REPO_DIR`) — kernel loads from disk after backend rebuild
2. Rebuild interval 4→5 segments (peak ~12GB, borderline)
3. First frame 580s → target 3-5 min

---

## 8. FAQ

**Q: The app is killed during generation (no crash dialog, process vanishes)**
A: ColorOS LMK killed it. Foreground app memory ceiling is ~8-9GB. Make sure precompiled ctxbin is used (§4.3) and segments rebuild every 4.

**Q: Error 6001 INVALID_HANDLE**
A: Graph operations must use handles from the graph's owning `QnnSet` (mixing global handles in dual-backend mode causes this). Also, the `GraphInfo` field order must match the SDK's `QnnWrapperUtils.hpp` (`{graph, graphName, inputTensors, ...}`).

**Q: Garbage/NaN output**
A: Check I/O dtype conversions use `--preserve_io datatype`; the transformer must be FP32 (FP16 overflows in QNN accumulation); feed raw latents to the VAE (the shiftfix graph already normalizes — don't re-normalize).

**Q: `libQnnHtp.so` transport 14001**
A: `ADSP_LIBRARY_PATH` must point to the dir containing `libQnnHtpV79Skel.so` (when validating via device CLI).

**Q: Third-party installer reinstall breaks generation**
A: Reinstall wipes the external data dir. Restore: 116 libs (incl. hexagon-v79 skel) + 8 assets + backend.txt=gpu + 12 ctxbin. `libQnnSystem.so` must be the full aarch64-android build (4,072,160 bytes) — the trimmed 217KB version makes all binaryInfo calls fail.

**Q: adb reports `no such file or directory`**
A: In git-bash, `export MSYS_NO_PATHCONV=1`, otherwise `/data/...` paths get MSYS-mangled.

**Q: Build fails after patching**
A: The patch tool converts line endings to CRLF; normalize with `tr -d '\r' < f > /tmp/x && mv /tmp/x f` (C++ files must be LF).

---

## 9. Project Structure

```
├── android/                      # Android Studio project
│   ├── app/src/main/
│   │   ├── java/com/example/zimage/
│   │   │   ├── MainActivity.kt   # M3 UI + state machine + generation dispatch
│   │   │   ├── HistoryStore.kt   # persistent history (timestamped PNG + JSON index)
│   │   │   ├── SettingsStore.kt  # sampling params persistence
│   │   │   ├── UiUtil.kt         # save/share/preview/clipboard/log export
│   │   │   └── ZImageRuntime.kt  # JNI wrapper
│   │   ├── cpp/
│   │   │   ├── zimage_runtime.cpp # ★ core: tokenizer→text→DiT→scheduler→VAE
│   │   │   ├── CMakeLists.txt
│   │   │   └── vndksupport.*      # QNN lib linking support
│   │   ├── assets/               # tokenizer vocab + GPU probe (in repo)
│   │   └── res/                  # M3 theme/layouts/icons/strings
│   ├── build.gradle / settings.gradle / gradle.properties
│   └── local.properties          # local SDK path (not in repo)
├── tools/                        # model export/quantize/convert/deploy scripts (46)
├── config/                       # ModelScope model manifest
├── docs/                         # bring-up logs, backend build, IO contract
├── PROJECT_HANDOFF.md            # ★ full handoff doc (all pitfalls & fixes)
├── README.md                     # this file (English)
└── README_cn.md                  # Chinese version
```

---

## 10. References

- `PROJECT_HANDOFF.md` — full handoff doc (architecture details, numeric contracts, every pitfall & fix)
- `docs/gpu-backend-build.md` — GPU backend v10 conversion→device validation, full command set
- `docs/backend-plan.md` — component plan
- `docs/qnn-v3-io-contract.md` — QNN I/O layout contract
- diffusers reference: `pipeline_z_image.py` + `scheduling_flow_match_euler_discrete.py`

---

## License / Disclaimer

This repository does **not** contain model weights or QNN SDK binaries (Qualcomm proprietary). Model weights are governed by Alibaba ModelScope/Tongyi-MAI licenses; the QNN SDK is governed by Qualcomm's license. Respect each license before use.
