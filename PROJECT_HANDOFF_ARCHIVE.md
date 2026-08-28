# Z-Image on Snapdragon 8 Elite - Project Handoff Document

## Project Overview

**Objective**: Adapt ModelScope's Z-Image-Turbo diffusion model to run on Snapdragon 8 Elite Android devices, targeting QNN HTP/NPU for text encoding and Adreno GPU for Transformer/VAE inference.

**Target Device**: Snapdragon 8 Elite (Adreno 830 GPU, Hexagon NPU)

**Model**: Tongyi-MAI/Z-Image-Turbo (ModelScope)
- Transformer: ~6.15B parameters, 30 layers, hidden 3840, 30 heads
- Text Encoder: Qwen3, ~4B parameters, 36 layers, hidden 2560
- VAE: ~84M parameters, latent channels=16

**Inference Target**:
- Resolution: 512x512
- Batch: 1
- Steps: 8 (Turbo)
- Latent shape: [1, 16, 64, 64]
- Conditioning shape: [1, 512, 2560]

**Backend Allocation**:
- Transformer: Adreno GPU (FP16 → INT4/FP16 mixed)
- Text Encoder: QNN HTP/NPU
- VAE: Adreno GPU (FP16)
- Scheduler: CPU (C++)

---

---
## Solution Update (2026-08-16, evening): HTP/NPU path is WORKING in the app sandbox

### Conclusion
- On the current device **PJZ110 / OnePlus 13 / SM8750 (SD 8 Elite)**, QNN **HTP/NPU**
  acceleration is **feasible from a normal app sandbox**. Verified end-to-end in-app:
  `deviceCreate=0`, `contextCreate=0`, `graphFinalize=0`, `QnnGraph_execute status 0x0`,
  VAE zero-latent probe returned `min=-0.421631 max=0.428467`.
- QNN **GPU/OpenCL remains NOT viable from the app sandbox** on this device:
  `libQnnGpu.so backendCreate=1006`, `clGetPlatformIDs=-1001`; vendor OpenCL cannot
  resolve its HAL/EGL dependencies in `clns-9` and cannot reach `/dev/kgsl-3d0`.
  Use **HTP for VAE, Transformer, and text encoder**.
- The same VAE model was also executed successfully from `adb shell` with both
  `qnn-net-run --backend libQnnHtp.so` and `--backend libQnnGpu.so`, which proves the
  hardware + QAIRT 2.49 libs are correct; only the GPU app-sandbox path is blocked.

### Root causes and final fixes (all verified on device)
1. **QnnHtp could not find `libQnnHtpV79Stub.so`**
   - QNN calls `dlopen("libQnnHtpV79Stub.so")` by bare SONAME. A file staged only in
     `files/zimage-runtime/lib/arm64-v8a/` is not on the app linker search path.
   - Fix: copy the stub into `build/android/apk-jniLibs/arm64-v8a/` so it is packaged
     into the APK native-lib directory. In JNI it can also be preloaded with
     `RTLD_NOW|RTLD_GLOBAL`.
2. **Do NOT preload the app-staged copy of vendor `libcdsprpc.so`**
   - `dlopen` of the copied `/vendor/lib64/libcdsprpc.so` fails in `clns-9` with:
     `library "libhidlbase.so" not found`.
   - Fix: keep `<uses-native-library android:name="libcdsprpc.so" .../>` in
     `AndroidManifest.xml` and let QNN load the **public vendor** `libcdsprpc.so`.
     On-device log confirms: `bare-dlopen-end libcdsprpc.so -> ok`.
3. **Hexagon skel library must be reachable via `ADSP_LIBRARY_PATH`**
   - Required file: `lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so` (12,038,356 bytes).
   - Deploy it to:
     `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/lib/arm64-v8a/`
     (Kotlin startup staging copies it into app-private storage automatically).
   - In JNI, before `QnnDevice_create`:
     `::setenv("ADSP_LIBRARY_PATH", (runtimeRoot + "/lib/arm64-v8a").c_str(), 1);`
4. **`QnnModel_composeGraphs` does NOT finalize the graph**
   - First in-app execute failed with status `0x1787`, QNN log:
     `Graph 256 was not Finalized`.
   - Fix: after `QnnModel_composeGraphs`/`graphRetrieve`, call
     `QnnGraph_finalize(graph, nullptr, nullptr)` and check status before execute.
5. Earlier fix remains required: `QnnDevice_create` first argument is
   `Qnn_LogHandle_t`, not the backend handle.

### Files changed this session
- `android/app/src/main/cpp/zimage_runtime.cpp`
  - safe `bareLoad()`/`preload()` diagnostics (no raw `dlerror()` in string concat;
    earlier one-line version caused a SIGSEGV at `std::string::append(nullptr)`),
  - HTP path: public-vendor `libcdsprpc.so` + `libQnnHtpV79Stub.so` global preload,
    `ADSP_LIBRARY_PATH` setup, explicit `QnnGraph_finalize` after compose.
  - backup: `android/app/src/main/cpp/zimage_runtime.cpp.bak_20260816`
- `android/app/src/main/java/com/example/zimage/MainActivity.kt`
  - runs one automatic zero-latent VAE feasibility probe right after runtime init.
- `build/android/apk-jniLibs/arm64-v8a/libQnnHtpV79Stub.so` (new APK JNI asset)
- Runtime staging on device: added `libQnnHtpV79Skel.so` under
  `<shared runtime>/lib/arm64-v8a/`.

### Current app status after fix
```text
System: loaded
GPU: loaded      (backendCreate=1006 -> fallback)
HTP: loaded      (backendCreate=0)
VAE library: loaded
VAE graph: graph=model input=latent[1,64,64,16] output=image[1,512,512,3]
VAE probe executed: min=-0.421631 max=0.428467
```

### Next steps
1. Replace the temporary auto-probe with the real pipeline wiring.
2. Re-convert/re-target VAE and Transformer to HTP (VAE model library itself is
   backend-agnostic; graph executes on whichever backend creates the context).
3. Fix and deploy int8 text-encoder segments for HTP.
4. Add text encoder -> Transformer -> VAE diffusion loop in JNI.

## Solution Update (2026-08-16, night): app-sandbox GPU alternative via LiteRT GPU delegate

### Answer to "any other way to use the GPU?"
- QNN `libQnnGpu.so` requires vendor OpenCL / `/dev/kgsl-3d0` and remains blocked in
  the app sandbox.
- Google AI Edge Gallery (`github.com/google-ai-edge/gallery`) shows the
  sandbox-safe route: it uses LiteRT / TFLite GPU delegate dependencies
  (`play-services-tflite-gpu 16.4.0`, LiteRT-LM `Backend.GPU()/Backend.NPU()`).
- LiteRT-LM Android AAR ships `libLiteRtClGlAccelerator.so`; its NEEDED list is
  public `libEGL.so` + `libGLESv3.so`, and it only optionally dlopens
  `libOpenCL`. LiteRT GPU delegate (`litert-gpu` AAR) has the same profile:
  NEEDED = `libEGL/libGLESv3`, optional OpenCL.
- Therefore GPU acceleration is possible from a normal Android app by moving the
  model to **TFLite/LiteRT + GPU delegate**, not by fighting the QNN OpenCL path.

### Implemented feasibility probe (verified on PJZ110 in app)
- Added `com.google.ai.edge.litert:litert:1.4.2` and
  `com.google.ai.edge.litert:litert-gpu:1.4.2` to `android/app/build.gradle`.
- Added `android/app/src/main/assets/gpu_probe.tflite`
  (10x10 float add model from LiteRT testdata).
- Added `android/app/src/main/java/com/example/zimage/GpuProbe.kt`:
  - direct native-order ByteBuffer model load,
  - `CompatibilityList().isDelegateSupportedOnThisDevice`,
  - `Interpreter.Options.addDelegate(GpuDelegate())`,
  - multi-input API `runForMultipleInputsOutputs(...)`.
- `MainActivity.kt` now runs this probe automatically after the QNN HTP VAE probe.
- On-device UI/log result:
  `LiteRT GPU probe: OK delegate=true outShape=10x10 out00=0.75 durationNs=797032`.

### Current backend verdict on this device
| Path | App sandbox | Notes |
|------|-------------|-------|
| QNN HTP/NPU | ✅ WORKING | VAE composed + finalized + executed on Hexagon |
| QNN GPU/OpenCL | ❌ blocked | backendCreate=1006, clGetPlatformIDs=-1001 |
| LiteRT GPU delegate (GLES/EGL) | ✅ WORKING for probe | full Z-Image model conversion still required |
| LiteRT-LM GPU/NPU | ✅ architecture | LLM-specific; not directly a diffusion runtime |

### Next steps if pursuing LiteRT GPU
1. Convert Z-Image components to TFLite/LiteRT, e.g. AI Edge Torch for
   PyTorch export, or ONNX -> TFLite where ops are supported.
2. Validate Transformer/VAE op coverage with `CompatibilityList` and run
   partitioned GPU delegates.
3. Keep text encoder on QNN HTP segments (already working direction).
4. Compare LiteRT GPU vs QNN HTP latency/memory and pick one backend per component.

## Solution Update (2026-08-16, late night): Transformer INT8/INT4 quantization attempt + bug fixes

### Bugs fixed this session
1. `tools/export_vae_onnx.py`
   - VAE export was missing `shift_factor`. Fixed to
     `latent / scaling_factor + shift_factor` before `vae.decode`.
   - VAE ONNX must be re-exported and re-converted to QNN before real-image use.
2. `tools/export_transformer_onnx.py`
   - Added fixed-shape `cap_mask[512]` input and patched
     `patchify_and_embed` / `_prepare_sequence` / `_build_unified_sequence`
     so padded prompt tokens are masked in attention, not just replaced by
     `cap_pad_token`.
   - New artifact: `build/transformer_fp16_capmask/transformer.onnx`
     (12 GB external weights, 4 inputs).
3. Text-encoder final norm clarification
   - Official pipeline uses `hidden_states[-2]` (layer 35 output before final
     RMSNorm). `final_norm.onnx` must NOT be used for conditioning.

### Transformer quantization findings
- Full-graph QAIRT quantization is NOT viable on this host/graph size:
  - W8A8 full: failed at `/transformer/layers.18/attention/to_q/MatMul`,
    `Invalid QnnModel constructed`.
  - W4A16 full: failed at `/transformer/layers.18/feed_forward/w3/MatMul`,
    `Tensor size is greater than available memory`.
- Full-graph FP16 conversion without quantization did complete but produced a
  **23 GB FP32 model.bin** (converter defaulted weights to FP32); discard it.
- **Solution: segment the 30 main DiT layers**, exactly like the text encoder.
  - `tools/export_transformer_segments.py` exports:
    `frontend` + 5 x 6-layer groups + `final`.
  - Artifacts: `build/transformer_segments_v2/` (~12 GB FP16 external data).
  - Segment I/O:
    `unified[1,1536,3840]`, `unified_freqs[1,1536,128]`,
    `unified_mask[1,1536]`, `adaln_input[1,256]`.

### Segmented quantization results (QAIRT converter + CPU quantizer runtime)
| Part | W8A8 model.bin | W4A8 model.bin |
|---|---|---|
| frontend | not run | 363,991,040 B (347 MB) |
| layers_00_05 | 1.1 GB | 543,713,280 B (519 MB) |
| other 4 groups | not run | 543,713,280 B each |
| final | not run | 1,269,760 B (1.2 MB) |
| **estimated total W4A8** | - | **~3.08 GB** |

- W4A8 uses `--weights_bitwidth 4 --act_bitwidth 8 --pack_4_bit_weights`.
- Conversion output: `build/qnn_transformer_segments_w4a8/{frontend,layers_* ,final}`.
- W8A8 single group is 1.1 GB; W4A8 is the recommended mobile setting
  (roughly half the weight footprint).

### Files added/changed this session
- Changed:
  - `tools/export_vae_onnx.py`
  - `tools/export_transformer_onnx.py`
  - `tools/convert_transformer_htp.ps1` (full-graph W8A8/W4A16 experiment)
- Added:
  - `tools/export_transformer_segments.py`
  - `tools/generate_transformer_calibration.py`
  - `tools/generate_transformer_segment_calibration.py`
  - `tools/convert_transformer_segments_htp.ps1`
  - `docs/backend-plan.md`
- Calibration raw data:
  - `E:\zimage_calib_transformer`
  - `E:\zimage_calib_transformer_layers`
  - `E:\zimage_calib_transformer_frontend`
  - `E:\zimage_calib_transformer_final`

### Android .so build validation
- Created `tools/build_transformer_qnn_libs.ps1`.
- Created no-space QNN junction `E:\qnn` -> QAIRT SDK root.
- Built and validated the W4A8 frontend Android library:
  `build/android/jniLibs/arm64-v8a/libqnn_transformer_frontend.so`
  = **364,921,888 bytes**.
- Remaining work: run the same script for `layers_00_05..24_29` and `final`.

### Next steps
1. Build remaining six W4A8 Transformer `.so` files.
2. JNI runtime: frontend -> 5 layer groups -> final, passing the four
   intermediate buffers; remember `noise_pred = -noise_pred` before scheduler.
3. Re-export and re-convert corrected VAE (shift_factor) as HTP FP16.
4. Deploy INT8 text-encoder `.so` files and wire the full
   tokenizer -> text encoder -> Transformer x8 -> VAE pipeline.



## Current Environment After Migration (2026-08-14)

- Project root: `E:\projects\zimage on phone\zimage on phone`
- Runtime root: `E:\projects\zimage on phone\zimage-runtime`
- QAIRT SDK: `E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730`
- QAIRT Python: `E:\projects\zimage on phone\zimage-runtime\python312\python.exe`
- Gradle cache: `E:\projects\zimage on phone\zimage-runtime\gradle-home`
- Android SDK: `C:\Users\Max\AppData\Local\Android\Sdk`
- Android NDK: `C:\Users\Max\AppData\Local\Android\Sdk\ndk\28.0.13004108`
- Java: `C:\Program Files\Android\Android Studio\jbr\bin\java.exe`
- Gradle launcher: `C:\Users\Max\.gradle\wrapper\dists\gradle-8.11.1-bin\bpt9gzteqjrbo1mjrsomdt32c\gradle-8.11.1\bin\gradle.bat`
- ADB: `C:\WINDOWS\system32\adb.exe`
- Device: `3B15B100YVR00000` (`PJZ110`, `arm64-v8a`), currently not connected during migration scan.
- Android model artifacts: `build\android\jniLibs\arm64-v8a`, six libraries, each 2,423,451,256 bytes.
- APK JNI assets remain lightweight; multi-GB model/runtime files are deployed to `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime` and staged into app-private storage at startup.


## Continuation Update (2026-08-16, device bring-up - HTP breakthrough & OpenCL dead-end)

Full session log: `docs/device-bringup-log-2026-08-15.md` (appended section 8).
**IMPORTANT CORRECTION**: the 2026-08-14 claim "VAE GPU graph: loaded" was a
misread — the device screenshot at that time showed `VAE graph: backendCreate=1006`.
The VAE graph has **never** successfully created until the HTP work below.

### Final root-cause verdict on 1006 (GPU/OpenCL path is a DEAD END in app sandbox)
1. `libQnnGpu.so` dlopens OpenCL at runtime (`clGetPlatformIDs`). The device's
   `/vendor/lib64/libOpenCL.so` is a KHR ICD loader; real impl is
   `/vendor/lib64/libOpenCL_adreno.so`; no `.icd` files exist on device.
2. We bundled the full OpenCL stack into the APK (libOpenCL.so, libOpenCL_adreno.so,
   libc++.so, libbase.so, libcutils.so + a **hand-built stub libvndksupport.so**
   exporting `android_load_sphal_library@LIBVNDKSUPPORT` with SONAME+version script).
   All libs now dlopen OK, **but** `clGetPlatformIDs` returns -1001 (CL_INVALID_PLATFORM)
   and direct `clIcdGetPlatformIDsKHR` returns -30.
3. Root cause: `libOpenCL_adreno.so` internally dlopens /vendor HAL libs
   (libCB.so, libgsl.so, libq3dtools_adreno.so; searches /system/vendor/lib64/egl)
   and needs /dev/kgsl-3d0 — none reachable from a non-root app (clns-9 namespace
   + SELinux). **OpenCL on Adreno is not viable from an app sandbox. Drop it.**

### BREAKTHROUGH: HTP (NPU) backend works in the app sandbox
- On-device probe in `zimage_runtime.cpp`: `HTP backendCreate=0` (SUCCESS) vs GPU 1006.
- Text encoder segments were already converted for HTP — this validates the project's
  NPU plan. VAE and Transformer should target **HTP**, not GPU.

### HTP device/context creation — current state (BLOCKED at fastRPC transport)
- `QnnDevice_create(logger, cfg, dev)` signature: **first arg is Qnn_LogHandle_t, NOT
  backend** (QnnInterface.h:463). Passing backend caused 11003 INVALID_ARGUMENT.
  Fix applied; now QNN internal logs show:
  `Failed to create transport instance: 4000` / `Transport layer setup failed: 14001`
- 4000 = QNN_BACKEND_ERROR_NOT_SUPPORTED: HTP needs **fastRPC transport** to the CDSP
  via `libcdsprpc.so` + write access to `/dev/fastrpc-cdsp`
  (`crw-rw-r-- system system`, SELinux `vendor_qdsp_device`).
- Attempted fixes (in order):
  1. `AndroidManifest.xml`: added `<uses-native-library android:name="libcdsprpc.so"`
     (and adsprpc/sdsprpc, required=false) INSIDE `<application>` (AAPT rejects it at
     manifest top level). Build OK but transport still failed.
  2. Pushed `/vendor/lib64/libcdsprpc.so` + `libadsprpc.so` (497 KB each) into staging
     `files/zimage-runtime/lib/arm64-v8a/` so QNN can dlopen them. **Not yet tested**
     at pause time — next step is force-stop + relaunch and read jni.log.
- If fastRPC still fails: the app likely needs device root (chmod/SELinux) or
  platform/system signature. **Fallback: CPU backend** (`libQnnCpu.so` is already in
  staging, no fastRPC/OpenCL needed) — use it to validate the full composeGraphs →
  graphExecute → VAE decode chain end-to-end, then revisit HTP.

### On-device logging (NEW, replaces screenshots)
- `zimage_runtime.cpp` now writes `<runtime root>/jni.log` (append) for every step:
  ICD setup, dlopen results, backendCreate/deviceCreate/contextCreate, status, generate.
  Read with: `adb shell "run-as com.example.zimage cat files/zimage-runtime/jni.log"`
- QNN log callback (`QnnLog_CreateFn_t`: `(fmt, level, ts, va_list)` — use vsnprintf,
  va_list cannot be reused) writes `[QNN]` lines into the same file.
- Rebuild/install/relaunch cycle: `gradle.bat assembleDebug` → `adb install -r` →
  `am force-stop` → `am start` → `sleep 30` → run-as cat.

### Build/package gotchas learned today
- `android/app/build.gradle` line 13: `jniLibs.srcDirs = [env QNN_ANDROID_JNILIBS ?:
  "${rootDir}/../build/android/apk-jniLibs"]` — bundled .so MUST go into
  `build/android/apk-jniLibs/arm64-v8a/`, NOT `src/main/jniLibs/` (ignored).
  Current bundled set there: libOpenCL.so, libOpenCL_adreno.so, libbase.so,
  libc++.so, libcutils.so, libvndksupport.so (stub).
- gradle 8.11.1 cache corruption (`Could not read workspace metadata` /
  groovy-dsl metadata.bin): `gradle --stop` then delete
  `C:\Users\Max\.gradle\caches\8.11.1` and rebuild.
- `CMakeLists.txt` include path needs BOTH `include` and `include/QNN` (QNN headers
  use `#include "QnnCommon.h"` style inside HTP subdirs).
- Use python (not patch tool) for edits that must stay LF; patch tool flips CRLF.

### Files changed this session (uncommitted)
- `android/app/src/main/cpp/zimage_runtime.cpp` — jni.log logging, QNN log callback,
  GPU→HTP fallback, HTP logCreate, deviceCreate logger-arg fix.
- `android/app/src/main/cpp/stub_vndksupport.c` + `vndksupport.map` — stub libvndksupport.
- `android/app/src/main/jniLibs/arm64-v8a/` — OpenCL stack + stub (see above).
- `android/app/src/main/AndroidManifest.xml` — uses-native-library for fastRPC libs.
- `android/app/src/main/cpp/CMakeLists.txt` — include/QNN added.
- `tools/convert_vae_htp.ps1` — VAE HTP conversion (2-stage: converter then
  qnn-context-binary-generator with --backend libQnnHtp.so). NOTE: qnn-onnx-converter
  in 2.49 does NOT accept `--backend`; backend selection happens in
  qnn-context-binary-generator.

### Quantization audit (2026-08-15, earlier in session)
- Deployed text encoder was float32 unquantized (6 x 2.26 GiB = 13.5 GiB) because
  QAIRT skipped quantization (`Skipping quantization, no input_list provided`).
- Calibration data generated (`E:\zimage_calib_v2`, 8 prompts chained forward),
  all 6 segments re-converted to int8 (model.bin 605,798,400 B each, ~578 MiB).
- **BLOCKED**: int8 Android .so build fails at ndk-build — `QnnInterface.h not found`
  because `QNN_SDK_ROOT` contains spaces and include flag splits. Fix candidates:
  copy `include/QNN` into no-space workdir, or junction `E:\qnn` → QAIRT root.
- Transformer ONNX: FP32 `build/transformer.onnx` verified (max diff 1.07e-4),
  FP16 `build/transformer_fp16.onnx` exported (12.3 GB). FP16 CPU verification
  showed NaN from PyTorch CPU FP16 (precision), ONNX side min=-4.32 max=4.64 sane.

### Next steps (priority order)
1. Relaunch app after fastRPC libs staged; read jni.log → does transport pass?
2. If not: try CPU backend path (load libQnnCpu.so instead of HTP) to prove the
   compose→execute chain works on-device.
3. If HTP needs root: document requirement; consider VAE CPU for validation.
4. Fix int8 segment ndk-build include path; rebuild 6 int8 .so; replace fp32 libs.
5. Transformer: convert fp16 ONNX with HTP backend (not GPU).

## Continuation Update (2026-08-15, device bring-up)

Full session log: `docs/device-bringup-log-2026-08-15.md`. Highlights:

- **Device connected and deployed**: PJZ110 (OnePlus 13, Snapdragon 8 Elite) via adb; APK installed, QNN runtime (103 libs, ~14 GB) verified on device under `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime`.
- **Bug 1 fixed**: adb-pushed libs were `shell:ext_data_rw` mode 770, invisible to the app uid → `stageRuntimeFiles()` silently skipped. Fixed with `adb shell chmod -R a+rX ...` (must be added to `deploy_android_runtime.ps1`). After fix: System/GPU/HTP/VAE library all `loaded`.
- **Bug 2 open**: `VAE graph: backendCreate=1006` = `QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED`. Device has no OpenCL `.icd` registration (`/vendor/etc/OpenCL/vendors/` missing), so the KHR loader in `/vendor/lib64/libOpenCL.so` finds no platforms. Workarounds tried: `OCL_ICD_VENDORS` env + local .icd (blocked: vendor deps `libcutils.so`/`libvndksupport.so` not in app namespace, and bare-name `dlopen("libOpenCL.so")` fails since it's only in vendor public.libraries.txt). In-progress: bundle libOpenCL.so + libOpenCL_adreno.so + libcutils.so + libvndksupport.so into APK jniLibs — **blocked on build.gradle `jniLibs.srcDirs` override** pointing to `build/android/apk-jniLibs`; the 4 libs were placed in `src/main/jniLibs/` but NOT packaged.
- **QUANTIZATION (major)**: audit revealed the deployed text encoder was **float32 unquantized** (6×2.26 GiB = 13.5 GiB) because QAIRT skipped quantization (`Skipping quantization, no input_list provided`). Generated real calibration data (8 prompts, chained forward, fp16 load to avoid OOM) and re-converted all 6 segments to **int8**: each model.bin now 605,798,400 bytes (~578 MiB), 1/4 of fp32. Verified: 506 tensors @8bit, 24 @32bit. Calibration files at `E:\zimage_calib_v2` (no-space path required by QAIRT input_list).
- **Files changed (uncommitted)**: `zimage_runtime.cpp` (ICD env + dlopen diagnostics, tag `zimage-jni`); new `src/main/jniLibs/arm64-v8a/{libOpenCL,libOpenCL_adreno,libcutils,libvndksupport}.so`; new `tools/generate_qwen3_calibration.py`, `tools/convert_qwen3_segments_int8.ps1`; patched `tools/build_qnn_android_libs.ps1` (Python tarfile extraction).
- **BLOCKED at pause**: Android .so build for int8 segments fails at ndk-build — `QnnInterface.h not found` because `QNN_SDK_ROOT` has spaces (`E:\projects\zimage on phone\...`) and the include flag splits. Fix candidates: copy `include/QNN` into the no-space workdir, or junction `E:\qnn` → QAIRT root. **Deployment to device deferred per user request.**
- **Next**: fix include path → build 6 int8 .so → replace fp32 libs on device → verify RSS ~3.5 GiB for text encoder.

## Continuation Update (2026-08-14)

- Fixed `tools/deploy_android_runtime.ps1`: removed literal PowerShell `` `r`n `` corruption; all three migration/build/deploy scripts parse successfully.
- Deployment now preserves `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime`, removes only obsolete `/data/local/tmp/zimage-runtime`, and skips libraries whose remote byte size already matches.
- Segmented text encoder artifacts are complete and verified: `build/text_encoder_segments_v2/manifest.json` covers 36 layers in six groups; each QNN group has `model.cpp`, `model.bin`, and `model_net.json`.
- Android JNI libraries are present under `build/android/jniLibs/arm64-v8a`: six `libqnn_layers_*.so` files, each 2,423,451,256 bytes.
- Migrated paths verified: QAIRT, Python 3.12, Gradle cache, Android SDK, and NDK 28.0.13004108 all exist.
- APK build verified successfully with Gradle 8.11.1 and Microsoft JDK 17.0.19 using `-Dorg.gradle.java.home=C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot`.
- Current device status: `adb devices` currently reports no connected device; deployment/install/runtime UI verification remains pending until `PJZ110` reconnects.
## Latest Update (2026-08-16, evening, phone offline — host-side build continues)

The device is temporarily unavailable, so work continued host-side per the current plan.
All model libraries are now built; the next device session can go straight to deployment +
numerical validation.

### Completed this session
1. **Transformer W4A8 .so set (old quantized-IO v2)**: all 7 libraries built
   (`build/android/jniLibs/arm64-v8a/libqnn_transformer_*.so`, 3,091,109,144 bytes total).
2. **Text encoder INT8 .so set (old quantized-IO)**: all 6 built
   (`build/android/jniLibs-text-int8/arm64-v8a/`, 3.40 GB total).
3. **VAE shift_factor fix is now fully productized**
   - Re-exported: `build/vae_decoder_shiftfix/vae_decoder.onnx` (198,289,874 B).
   - ONNX vs PyTorch random-latent parity: max abs diff `7.32e-5`.
   - QNN FP16 conversion: `build/qnn_vae_shiftfix_fp16/` (`vae.bin` 99,174,400 B,
     FLOAT_16 tensors 314 / FLOAT_32 31).
   - Android lib: `build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so`
     (99,965,504 B); `zimage_runtime.cpp` now loads `libqnn_vae_shiftfix.so`
     first and falls back to `libqnn_vae_gpu.so`. APK rebuild verified.
4. **Discovered and fixed a critical segmented-quantization bug**
   - v2 per-segment calibration fed every 6-layer group independent random
     inputs (std 0.02). Chained ONNX runs show real boundaries grow to
     min/max of ~-105/+6236 while the old `unified_in` u8 encodings only
     represented ~+/-0.1 -> values were clipped massively at every boundary.
   - New tool `tools/generate_transformer_chain_calibration.py` runs the real
     FP16 ONNX chain and writes boundary-consistent calibration under
     `E:\zimage_calib_transformer_chain` (frontend -> 5 groups -> final).
   - Final chosen fix: reconvert all segments with
     `--preserve_io datatype`. Graph inputs/outputs stay FLOAT_16/FLOAT_32
     (the graph inserts its own Convert nodes), so **no CPU dequant/requant is
     needed at segment boundaries** while weights/activations remain W4A8.
5. **Transformer v3 (float-IO W4A8) converted and built**
   - QNN: `build/qnn_transformer_segments_v3_floatio_w4a8/` (~3,083,212,800 B).
   - Android .so: `build/android/jniLibs-transformer-v3/arm64-v8a/`
     (frontend 364,928,320 B + five groups 544,917,x B + final 1,029,120 B).
   - Scripts: `tools/convert_transformer_segments_v3.ps1`.
   - Calibration: frontend `E:\zimage_calib_transformer_frontend_bool`
     (cap_mask stored as bool/u8), layers/final `E:\zimage_calib_transformer_chain`.
   - **Important**: the QNN converter constant-folds and drops the frontend
     `unified_freqs` output. It is constant across inputs (verified samples
     00/01). Precomputed QNN-layout float32 asset:
     `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw`
     (dims `[1,128,1536]`, 786,432 B). Deploy it to
     `<runtime-root>/assets/unified_freqs_f32_nhwc.raw`.
6. **Text encoder INT8 v3 (float-IO) converted and built**
   - Calibration copied to no-space `E:\zimage_calib_qwen3_v2` (QAIRT
     input_list parsing splits paths on spaces).
   - QNN: `build/qnn_text_encoder_segments_int8_floatio/` (six x 605,798,400 B).
   - Android .so: `build/android/jniLibs-text-int8-floatio/arm64-v8a/`
     (six x ~606.7 MB, 3.40 GB total).
   - Graph I/O is now FLOAT_32 (dims follow QNN layout:
     `hidden_states[1,2560,512]`, `attention_mask[1,512,512,1]`,
     `cos/sin[1,128,512]`).
   - Script: `tools/convert_qwen3_segments_int8_floatio.ps1`.
7. **JNI Transformer v3 probe added and host-compiled**
   - `zimage_runtime.cpp`: `SegmentGraph`, `composeSegment`, `runGraph`,
     fp16<->fp32 conversion, `ensureTransformer`, and
     `Java_..._nativeTransformerProbe(latent, timestep, capFeats, capMask)`.
   - Executes frontend -> 5 layer groups -> final with QNN-layout buffers;
     loads `unified_freqs` from `assets/unified_freqs_f32_nhwc.raw`; reports
     per-segment ms and min/max/mean. Noise negation before scheduler is
     intentionally NOT applied (probe compares against ONNX `noise_pred`).
   - Kotlin `ZImageRuntime.transformerProbe(...)` added; APK `assembleDebug`
     succeeds.
8. **QNN manifest generator**
   - `tools/generate_qnn_manifest.py` emits JSON graph-IO contracts from
     `model_net.json`; v3 manifests in `build/qnn_manifest_v3/`.
   - Human-readable runtime contract: `docs/qnn-v3-io-contract.md`
     (segment names, QNN layouts, dtypes, assets, expected reference stats).
9. **Qwen3 embedding segment and text-encoder JNI probe**
   - QNN FP16 embedding: `build/qnn_text_encoder_embedding_fp16/embedding/`
     (model.bin 777,922,560 B), Android lib
     `build/android/jniLibs-text-int8-floatio/arm64-v8a/libqnn_embedding.so`
     (778,267,512 B). Graph IO: `input_ids[1,512]` INT_64 ->
     `hidden_states[1,512,2560]` FLOAT_32.
   - Precomputed text-encoder assets: `build/text_encoder_precomp/`
     `cos_qnn_f32.raw`/`sin_qnn_f32.raw` (`[1,128,512]`),
     `causal_mask_qnn_f32.raw` (`[1,512,512,1]`).
   - JNI `nativeTextEncoderProbe(inputIds, attentionMask)` added: composes /
     executes / frees segments one at a time (peak memory = one segment),
     embedding -> six INT8 float-IO groups, uses QNN layouts and skips
     `final_norm`; APK compiles.

### Deployment / validation queue for next device session
1. Push v3 model .so sets (transformer + text int8 float-io + embedding) and
   `libqnn_vae_shiftfix.so` to the runtime lib dir (skip old quantized-IO sets).
2. Push `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw` and
   `build/text_encoder_precomp/{cos,sin,causal_mask}_qnn_f32.raw` to
   `<runtime-root>/assets/` (`tools/deploy_android_runtime.ps1` now does this).
3. Install rebuilt APK; check status shows `libqnn_vae_shiftfix.so`.
4. Run `transformerProbe` with `E:\zimage_calib_transformer_frontend`
   sample_00 raw inputs and compare noise stats against the saved ONNX chain
   `E:\zimage_calib_transformer_chaininal\sample_00_noise_pred.raw`.
5. Then wire text encoder v3 sequentially, CPU tokenizer/embedding, scheduler
   and `noise_pred = -noise_pred` for the full pipeline.

## Disk Cleanup (2026-08-16 night)

Freed E: from ~87 GB available to ~400 GB. Deleted (all regenerable/replaced):
- Old calibration dirs (`zimage_calib_v2`, `zimage_calib_transformer*` old/layers/final/frontend/bool/chain_clip).
- Old build artifacts: `transformer_fp16.bin/capmask`, `text_encoder_opset17`, `text_encoder_segments`,
  `qnn_text_encoder_segments` (+int8), `qnn_transformer_segments_{w4a8,w8a8,v3,v4,chain_*}`,
  old `vae.onnx`/`qnn_vae_gpu`.
- 455 top-level `onnx__MatMul_*` temp files (~32 GB).
- Two `qnn_android_build` work dirs (36 GB + 68 GB), `tmp_5749` (6.8 GB).
- `zimage-runtime/qairt-tmp/tmp*` (27 GB), old `zimage-runtime/build` (5.5 GB).
- Duplicate model snapshot `zimage-runtime/models` (31 GB); keep `tools/models` (active model source).
Kept: v5 no-pack transformer, text v3 float-io + embedding, VAE shiftfix, active ONNX segment sources
(`text_encoder_segments_v2`, `transformer_segments_v2`, `transformer_segments_v3_floatmask`),
current android jniLibs, freqs/precomp assets, manifests, and `tools/models` (do NOT delete).

## Device Bring-up Update (2026-08-16 night, HTP v3/v5 validation — IN PROGRESS)

Device `3B15B100YVR00000` reconnected. v3/v5 model libs deployed, rebuilt APK installed,
probes run from raw files pushed to `<external>/probe/`. Current state is **partially
working**; text-encoder path looks good, Transformer path has an HTP packing/OOM issue
still being worked.

### 1. VAE corrected FP16 — WORKS on HTP ✅
- `zimage_runtime.cpp` now always initializes HTP (no GPU fallback).
- `libqnn_vae_shiftfix.so` loads (graph name `vae`).
- `graphFinalize=0`, zero-latent probe executed:
  `min=-0.538574 max=0.330078 mean=0.00953321` (finite; old GPU lib gave NaN).
- Graph name is already unique (`vae`), so it coexists with segment graphs.

### 2. Qwen3 v3 float-IO INT8 — segments execute on HTP ✅ (chain partially observed)
- `libqnn_embedding.so` + six `libqnn_layers_*.so` deployed (float-IO W8A8).
- Observed on-device HTP results:
  - `te_embedding` graph composed.
  - `te_layers_00_05`, `te_layers_06_11`, `te_layers_12_17` executed:
    `Graph ... execution finished with result 0`.
  - `te_layers_18_23` compose/execute was in progress when the Transformer probe
    OOM-killed the app on an earlier run; full 6-segment chain still needs one clean run.
- Calibration assets `cos/sin/causal_mask_qnn_f32.raw` staged and used.

### 3. Transformer — three separate blockers found and mostly fixed
1. **Packed W4A8 rejected by HTP v79**
   - `QNN_DATATYPE_UFIXED_POINT_4` graph tensor → `tensor datatype 1028 not supported`,
     `QnnTensor_createGraphTensor status 0x1b5c`, `compose=1`.
   - Fix: convert W4A8 **without** `--pack_4_bit_weights`; weights stay in
     `UFIXED_POINT_8` containers (v5).
2. **BOOL_8 StridedSlice rejected by HTP**
   - `Unsupported input/output datatypes ... op '_Slice_7'`, error `0xc26` / `3110`.
   - Root cause: `cap_mask`/padding mask was bool before Python slicing, so ONNX
     emitted bool `Slice`.
   - Fix: `tools/export_transformer_segments.py` now casts masks to float and uses
     `1.0 - mask` **before** slicing; new frontend ONNX at
     `build/transformer_segments_v3_floatmask/frontend/model.onnx` (cap_mask is float32).
     Re-converted to `build/qnn_transformer_segments_v5_w4a8_nopack/`.
3. **OOM when running segments sequentially**
   - v5 no-pack frontend composes; `tf_layers_00_05` and `tf_layers_06_11` executed on
     HTP with `result 0`.
   - During `tf_layers_12_17` compose the app was killed: `reason=3 (LOW_MEMORY)`,
     RSS ~11 GB. Root cause: QNN graph handles cannot be freed individually
     (see `QnnCommon.h`); only `contextFree` releases graphs, so sequential
     compose/execute/free accumulates prepared graphs in the one HTP context.
   - Planned fix (in progress): pass
     `QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING = true` as a graph config so the
     4-bit-range `UFIXED_POINT_8` weights are stored packed in the context binary,
     roughly halving weight memory.
   - **Current bug to fix next**: that graph-config wiring makes every
     `QnnModel_composeGraphs` return `8` (`MODEL_INVALID_ARGUMENT_ERROR`), including
     embedding/VAE-independent segments. Need to fix `GraphConfigInfo` /
     `QnnGraph_Config_t` construction (custom config pointer / graph-name matching),
     or revert the config and instead create+free a fresh QNN context per segment.

### 4. QNN model .so loader fixes found on-device (all applied to build scripts)
- objcopy must see relative path `obj/binary/<file>.raw`; an absolute Windows path
  produced `_binary_E__qnn_android_build_...` symbols and dlopen failed with
  `_binary_obj_binary_*` unresolved. Fixed in both `build_qnn_android_libs.ps1`
  and `build_transformer_qnn_libs.ps1`.
- `linux/QnnModelPal.cpp` was missing from builds → unresolved
  `qnn_wrapper_api::strnDup`. Now copied into every model workdir.
- QnnModel wrapper uses graph name `"model"`; duplicate names in one context fail
  with `Graph name is a duplicate`. v3/v5 model.cpp files are patched to unique
  names `tf_*` / `te_*`.
- `stageRuntimeFiles` now also copies `assets/` from external storage to the
  private runtime root.

### 5. Current artifact matrix
| Component | QNN model | Android .so | Device status |
|---|---|---|---|
| VAE corrected FP16 | `build/qnn_vae_shiftfix_fp16/` | `libqnn_vae_shiftfix.so` 99,919,104 B | ✅ HTP executed |
| Text encoder embedding FP16 | `build/qnn_text_encoder_embedding_fp16/` | `libqnn_embedding.so` 778,267,288 B | ✅ composed |
| Text encoder v3 W8A8 x6 | `build/qnn_text_encoder_segments_int8_floatio/` | `libqnn_layers_*.so` ~606.68 MB each | ✅ segments 0-2 executed, rest pending clean run |
| Transformer v5 W4A8 no-pack float-IO | `build/qnn_transformer_segments_v5_w4a8_nopack/` | `build/android/jniLibs-transformer-v5-nopack/` (frontend 728 MB + 5x1.087 GB + final 1.64 MB) | ⚠️ frontend + 2 layer groups executed, then OOM |
| Transformer v3 packed W4A8 float-IO | `build/qnn_transformer_segments_v3_floatio_w4a8/` | `build/android/jniLibs-transformer-v3/` | ❌ HTP rejects UFIXED_POINT_4 |
| freqs / text precomp assets | `build/transformer_freqs_qnn/`, `build/text_encoder_precomp/` | staged under `<runtime>/assets/` | ✅ |

### 6. Next steps after pause
1. Fix HTP `WEIGHTS_PACKING` graph config (or revert to no graph config) and
   re-run Transformer probe while watching RSS/logcat.
2. If packing does not reduce RSS enough: create a fresh QNN context for each
   segment (contextFree releases all graphs), accepting longer per-step setup, or
   evaluate context-binary caching for the 8-step loop.
3. Complete one clean text-encoder chain (embedding -> layers_30_35) and compare
   `hidden_states[-2]` against Python/ONNX reference.
4. Continue Transformer chain: frontend -> 5 groups -> final vs
   `E:/zimage_calib_transformer_chain/final/sample_00_noise_pred.raw`.
5. Then add `noise_pred = -noise_pred`, scheduler loop, and full pipeline.

## Current Status

### ✅ Completed

1. **Model Download**
   - Source: ModelScope (Tongyi-MAI/Z-Image-Turbo, revision: master)
   - Location: `E:\projects\zimage on phone\zimage-runtime\models\z-image-turbo\models\Tongyi-MAI--Z-Image-Turbo\snapshots\master`
   - Structure: `assets/`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`, `model_index.json`

2. **VAE Export**
   - Format: ONNX (opset 17)
   - Path: `E:\projects\zimage on phone\zimage-runtime\build\vae_decoder.onnx`
   - Shape: [1, 16, 64, 64] → [1, 3, 512, 512]
   - Status: ✅ Verified with ONNX Runtime CPU

3. **Text Encoder Export**
   - Format: ONNX (opset 17, fixed shape)
   - Path: `E:\projects\zimage on phone\zimage-runtime\build\text_encoder_opset17\`
   - Shape: [1, 512] → [1, 512, 2560]
   - Status: ️ QAIRT full-graph conversion fails (MATMUL_TO_FC optimization issue)
   - Next: Segmented export approach (layer groups)

4. **QAIRT SDK Installation**
   - Version: 2.49.0.260730
   - Location: `E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730`
   - Key tools: `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`
   - Android libs: `libQnnHtp.so`, `libQnnGpu.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`

5. **Independent Python 3.12 Runtime**
   - Location: `E:\projects\zimage on phone\zimage-runtime\python312`
   - Purpose: QAIRT conversion without affecting system Python 3.14
   - Installed packages: numpy, onnx==1.18.0, onnxruntime, protobuf, PyYAML, requests, packaging, psutil, setuptools, pandas

6. **Environment Migration**
   - All large artifacts moved to `E:\projects\zimage on phone\zimage-runtime`
   - Gradle cache: `E:\projects\zimage on phone\zimage-runtime\gradle-home`
   - QAIRT temp: `E:\projects\zimage on phone\zimage-runtime\qairt-tmp`
   - Original C:/D: files preserved (not deleted)

7. **Android Project Skeleton**
   - Location: `E:\projects\zimage on phone\zimage on phone\android`
   - Language: Kotlin + JNI/C++
   - AGP: 8.5.2
   - compileSdk: 36
   - targetSdk: 36 (Android 16)
   - minSdk: 30
   - NDK: 29.0.14206865
   - CMake: 3.31.6

8. **JNI Runtime**
   - File: `android/app/src/main/cpp/zimage_runtime.cpp`
   - Features: QNN backend dynamic loading (System, GPU, HTP)
   - Status: ✅ Loads successfully, reports backend availability

9. **APK Build**
   - Debug APK: `android/app/build/outputs/apk/debug/app-debug.apk`
   - Size: ~103 MB
   - Status: ✅ Builds successfully, runs on device without crash
   - Current UI: Centered layout with status text and "Generate test image" button
   - Button action: Shows "Runtime scaffold only: 512x512, 8 steps. Next: attach Transformer GPU backend."

10. **Android 16 Compatibility**
    - targetSdk 36
    - Edge-to-edge with system bar insets
    - Content vertically/horizontally centered

---

## Environment Configuration

### Paths

```
# Main workspace
E:\projects\zimage on phone\zimage-runtime

# Android project
E:\projects\zimage on phone\zimage on phone\android

# QAIRT SDK
E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730

# Python 3.12 (QAIRT)
E:\projects\zimage on phone\zimage-runtime\python312\python.exe

# Gradle cache
E:\projects\zimage on phone\zimage-runtime\gradle-home

# Android SDK
C:\Users\Max\AppData\Local\Android\Sdk

# Java
C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot

# Gradle executable
C:\Users\Max\.gradle\wrapper\dists\gradle-8.11.1-bin\bpt9gzteqjrbo1mjrsomdt32c\gradle-8.11.1\bin\gradle.bat
```

### Environment Variables

```powershell
$env:QNN_SDK_ROOT = "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730"
$env:QAIRT_PYTHON = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe"
$env:QAIRT_TMP_DIR = "E:\projects\zimage on phone\zimage-runtime\qairt-tmp"
$env:ZIMAGE_MODEL_ROOT = "E:\projects\zimage on phone\zimage-runtime\models\z-image-turbo\models\Tongyi-MAI--Z-Image-Turbo\snapshots\master"
$env:GRADLE_USER_HOME = "E:\projects\zimage on phone\zimage-runtime\gradle-home"
$env:ANDROID_SDK_ROOT = "C:\Users\Max\AppData\Local\Android\Sdk"
$env:ANDROID_HOME = $env:ANDROID_SDK_ROOT
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"
```

### Build Command

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\tools\setup_qairt_env.ps1

$env:GRADLE_USER_HOME = "E:\projects\zimage on phone\zimage-runtime\gradle-home"
$env:ANDROID_SDK_ROOT = "C:\Users\Max\AppData\Local\Android\Sdk"
$env:ANDROID_HOME = $env:ANDROID_SDK_ROOT
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:ANDROID_SDK_ROOT\platform-tools;$env:PATH"

$g = "C:\Users\Max\.gradle\wrapper\dists\gradle-8.11.1-bin\bpt9gzteqjrbo1mjrsomdt32c\gradle-8.11.1\bin\gradle.bat"
& $g :app:assembleDebug --no-daemon --console=plain
```

---

## Known Issues

### 1. QAIRT Text Encoder Conversion Failure

**Problem**: QAIRT 2.49 fails during full-graph conversion of Qwen3 ONNX with `MATMUL_TO_FC` and `ALIGN_MATMUL_RANKS` optimization errors.

**Root Cause**: QAIRT's forced MatMul optimization is unstable with large external weight sets (~15GB, 326 files).

**Attempted Fixes**:
- Opset 17
- Folded constant `LessOrEqual` mask node
- Merged external data into single sidecar
- `--target_backend LPAI`
- `--no_simplification`
- Moved temp directory to D: (now E:)

**Next Approach**: Segmented export - split Qwen3 into fixed layer groups, compile each as smaller QNN graph.

### 2. AGP compileSdk Warning

**Problem**: Android Gradle Plugin 8.5.2 was tested up to compileSdk 34, but we use 36.

**Impact**: Warning only, does not block build.

**Fix**: Update AGP to newer version when available, or suppress warning in `gradle.properties`:
```properties
android.suppressUnsupportedCompileSdk=36
```

### 3. Gradle Cache Corruption

**Problem**: Original `C:\Users\Max\.gradle\caches` had corrupted `metadata.bin` files causing build failures.

**Fix**: Migrated to fresh `E:\projects\zimage on phone\zimage-runtime\gradle-home`.

---

## Pending Work

### Phase 1: Text Encoder Segmented Export

**Goal**: Split Qwen3 into manageable layer groups for QNN conversion.

**Steps**:
1. Create new export script: `tools/export_text_encoder_segmented.py`
2. Export strategy:
   - Embedding layer (fixed)
   - Layers 0-11 (group 1)
   - Layers 12-23 (group 2)
   - Layers 24-35 (group 3)
   - Final norm (fixed)
3. Each segment: separate ONNX with fixed input/output shapes
4. Convert each segment with QAIRT independently
5. Android runtime: load segments sequentially, pass hidden states between them

**Estimated Effort**: 2-3 days

### Phase 2: Transformer GPU Backend

**Goal**: Attach Adreno GPU backend for Transformer inference.

**Steps**:
1. Export Transformer to ONNX (opset 17, fixed shape)
   - Input: latent [1, 16, 64, 64], conditioning [1, 512, 2560], timestep [1]
   - Output: predicted noise/latent [1, 16, 64, 64]
2. Convert to QNN GPU backend using `qnn-onnx-converter --target_backend GPU`
3. Generate context binary with `qnn-context-binary-generator`
4. Integrate into Android JNI runtime
5. Implement 8-step diffusion loop in C++

**Challenges**:
- Transformer is 6.15B parameters - memory management critical
- May need weight-only INT4 quantization
- Adreno GPU FP16 first, then mixed precision

**Estimated Effort**: 3-5 days

### Phase 3: Complete Inference Pipeline

**Goal**: End-to-end image generation on device.

**Steps**:
1. CPU tokenizer (already exported: `build/text-inputs.json`)
2. Text encoder: run segmented QNN graphs sequentially
3. Transformer: run 8 diffusion steps on GPU
4. VAE decoder: run on GPU
5. Scheduler: C++ CPU implementation (DDPM/DDIM)
6. Output: RGB image [512, 512, 3]

**Memory Budget** (target):
- Text encoder: ~2GB (segmented, quantized)
- Transformer: ~3-4GB (INT4 weights, FP16 activations)
- VAE: ~200MB (FP16)
- Total: ~5-6GB peak

**Estimated Effort**: 3-4 days

### Phase 4: Optimization

**Goal**: Reduce latency and memory usage.

**Steps**:
1. Text encoder: INT4 quantization, KV cache (if needed)
2. Transformer: weight-only INT4, activation FP16
3. VAE: already FP16, verify numerical parity
4. Graph fusion opportunities
5. Memory pooling across components
6. Profile with QNN profiling tools

**Estimated Effort**: 2-3 days

---

## File Structure

```
E:\projects\zimage on phone\zimage-runtime\
├── qairt\                          # QAIRT SDK 2.49.0.260730
│   └── qairt\2.49.0.260730\
│       ├── bin\
│       │   ├── x86_64-windows-msvc\
│       │   └── aarch64-android\
│       ├── lib\
│       │   ├── python\             # QAIRT Python modules
│       │   └── aarch64-android\    # QNN .so files
│       ── include\QNN\            # C headers
├── python312\                      # Independent Python 3.12 runtime
├── models\                         # ModelScope Z-Image-Turbo
│   └── z-image-turbo\models\Tongyi-MAI--Z-Image-Turbo\snapshots\master\
│       ├── text_encoder\           # Qwen3 safetensors (3 files, ~8GB)
│       ├── transformer\            # DiT safetensors
│       ├── vae\                    # VAE safetensors
│       ├── tokenizer\
│       ├── scheduler\
│       └── model_index.json
├── build\                          # Exported ONNX models
│   ├── vae_decoder.onnx            # ✅ Verified
│   ├── text_encoder_opset17\       # ️ Needs segmented conversion
│   └── text-inputs.json            # Tokenizer test inputs
├── qnn\                            # QNN conversion attempts
│   └── onnx\
│       └── text_encoder_qnn_consolidated.onnx + .data (5.6GB)
├── android\                        # Android deployment libs
│   └── jniLibs\arm64-v8a\
│       ├── libQnnGpu.so
│       ├── libQnnHtp.so
│       ├── libQnnHtpPrepare.so
│       └── libQnnSystem.so
├── qairt-tmp\                      # QAIRT temporary files
└── gradle-home\                    # Gradle cache (fresh)

E:\projects\zimage on phone\zimage on phone\
├── android\                        # Android Studio project
│   ├── app\
│   │   ├── build.gradle            # AGP 8.5.2, compileSdk 36, NDK 29
│   │   └── src\main\
│   │       ├── cpp\
│   │       │   ├── CMakeLists.txt
│   │       │   └── zimage_runtime.cpp  # JNI with QNN loading
│   │       ├── java\com\example\zimage\
│   │       │   ├── MainActivity.kt     # Centered UI, Android 16 safe
│   │       │   └── ZImageRuntime.kt    # JNI wrapper
│   │       └── AndroidManifest.xml
│   └── build.gradle                # Root build config
├── tools\
│   ├── setup_qairt_env.ps1         # QAIRT environment setup
│   ├── run_qairt_python312.ps1     # Run Python with QAIRT
│   ├── compile_qnn_htp.ps1         # QNN conversion script
│   ├── deploy_android_runtime.ps1  # Push libs to device
│   ├── export_vae_onnx.py          # VAE export (completed)
│   ├── export_text_encoder_onnx.py # Text encoder export (completed)
│   ├── prepare_text_inputs.py      # Tokenizer test inputs
│   ├── verify_onnx.py              # ONNX verification
│   └── consolidate_onnx_external_data.py  # Merge external weights
├── config\
│   └── modelscope-z-image-turbo.json  # Model config and targets
├── docs\
│   └── bring-up.md                 # Milestone documentation
└── README.md                       # Project overview
```

---

## Key Scripts

### QAIRT Environment Setup

```powershell
# tools/setup_qairt_env.ps1
param([string]$QnnSdkRoot = $(if ($env:QNN_SDK_ROOT) { $env:QNN_SDK_ROOT } else { "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730" }))
if (-not (Test-Path (Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"))) { throw "Invalid QAIRT root: $QnnSdkRoot" }
$env:QNN_SDK_ROOT = (Resolve-Path $QnnSdkRoot).Path
$env:Path = "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc;$env:Path"
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python;$env:PYTHONPATH"
$env:QAIRT_PYTHON = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }
New-Item -ItemType Directory -Force $env:QAIRT_TMP_DIR | Out-Null
```

### Android Deployment

```powershell
# tools/deploy_android_runtime.ps1
param(
    [string]$Adb = "adb",
    [string]$Device = "",
    [string]$RuntimeRoot = "E:\projects\zimage on phone\zimage-runtime",
    [string]$RemoteRoot = "/data/local/tmp/zimage-runtime",
    [switch]$IncludeModel
)

# Push QNN libs to device
$libDir = Join-Path $RuntimeRoot "android\jniLibs\arm64-v8a"
Get-ChildItem $libDir -Filter "*.so" | ForEach-Object {
    & $Adb push $_.FullName "$RemoteRoot/lib/arm64-v8a/$($_.Name)"
}

# Optionally push model
if ($IncludeModel) {
    $model = Join-Path $RuntimeRoot "models\z-image-turbo\models\Tongyi-MAI--Z-Image-Turbo\snapshots\master"
    & $Adb push $model "$RemoteRoot/models/"
}
```

---

## Next Immediate Steps

1. **Text Encoder Segmented Export**
   - Create `tools/export_text_encoder_segmented.py`
   - Export 4 segments: embedding, layers 0-11, 12-23, 24-35, final norm
   - Test each segment with ONNX Runtime
   - Attempt QAIRT conversion on each segment independently

2. **Transformer Export**
   - Create `tools/export_transformer_onnx.py`
   - Fixed shape: latent [1,16,64,64], conditioning [1,512,2560], timestep [1]
   - Export to ONNX opset 17
   - Verify with ONNX Runtime CPU

3. **JNI Integration**
   - Extend `zimage_runtime.cpp` with QNN backend initialization
   - Load text encoder segments sequentially
   - Load Transformer GPU context binary
   - Implement diffusion loop skeleton

---

## Reference Information

### ModelScope Download

```python
# tools/download_model.py
from modelscope import snapshot_download
snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='E:/zimage-runtime/models/z-image-turbo')
```

### QAIRT Conversion Command

```bash
# Text encoder (segmented approach needed)
qnn-onnx-converter \
  --input_network text_encoder_segment.onnx \
  --output_path text_encoder_segment.cpp \
  --input_dim input_ids 1,512 \
  --input_dim attention_mask 1,512 \
  --target_backend LPAI

# Context binary generation (Android)
qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --model text_encoder_segment.cpp \
  --binary_file text_encoder_segment.bin
```

### Android NDK Build

```bash
# CMake configuration
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -B build \
  -S .

# Build
cmake --build build --config Release
```

---

## Contact & Resources

- **ModelScope**: https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo
- **QAIRT Documentation**: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
- **QNN SDK API**: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
- **Snapdragon 8 Elite Specs**: Adreno 830 GPU, Hexagon NPU (v79)

---

## Version History

- **2026-08-10**: Initial project setup, VAE export, text encoder export, QAIRT installation, Android skeleton, APK build, environment migration to E:
- **2026-08-14**: Text encoder segmented export completed (all 6 QNN groups), VAE GPU conversion + Android library, JNI dynamic QNN loading, VAE loader fix, device verification
- **2026-08-15**: Transformer ONNX export (FP32 verified, FP16 done), RoPE real-valued patch, QAIRT conversion script; quantization audit (text encoder was fp32 → int8 re-conversion with calibration)
- **2026-08-16**: On-device bring-up: OpenCL/GPU path proven dead-end in app sandbox; HTP backendCreate SUCCESS; fastRPC transport blocked (SELinux /dev/fastrpc-cdsp); jni.log file logging; stub libvndksupport; fastRPC libs staged

---

## Notes for Next AI

1. **Do NOT delete original C:/D: files** - they are preserved as backup
2. **Use E: for all large artifacts** - C: has limited space
3. **Text encoder full-graph conversion fails** - use segmented approach
4. **Gradle cache is at E:\projects\zimage on phone\zimage-runtime\gradle-home** - do not use C:\Users\Max\.gradle
5. **targetSdk 36 requires edge-to-edge handling** - already implemented
6. **QNN libs are loaded dynamically** - check `dlopen` results in JNI
7. **APK is debug-signed** - for production, configure release signing

---

**Last Updated**: 2026-08-16 (night, update 4 — on-device HTP validation in progress)
**Status**: Device connected and v3/v5 libs deployed. Working: corrected VAE FP16 executes on HTP (finite output); Qwen3 v3 float-IO W8A8 embedding + layer groups execute on HTP (chain partially observed). Blocked mid-fix: packed Transformer W4A8 is rejected by HTP (UFIXED_POINT_4 unsupported), so v5 no-pack W4A8 is used; frontend + layers 00-05 + 06-11 executed, then app was LOW_MEMORY killed (RSS ~11 GB) because QNN graph handles cannot be freed individually. Current APK has a broken-in-progress HTP WEIGHTS_PACKING graph config (compose returns 8) that must be fixed next; see Device Bring-up Update section.

## Device Verification Update (2026-08-14)

- Device reconnected: `3B15B100YVR00000` / `PJZ110`, `arm64-v8a`.
- Incremental deployment completed. All six model segment libraries and QAIRT libraries were already present with matching sizes; no large-file retransmission was needed. `/data/local/tmp/zimage-runtime` is absent.
- APK installed successfully and `com.example.zimage/.MainActivity` is foreground.
- `MANAGE_EXTERNAL_STORAGE` app-op: `allow`.
- UI status verified on device: `System: loaded`, `GPU: loaded`, `HTP: loaded`.
- Test button invocation verified: JNI returned `Runtime scaffold only: 512x512, 8 steps. Next: attach Transformer GPU backend.`
- Shared runtime directory verified at `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime`; size approximately 14G with 102 library files.
- No `AndroidRuntime`, `FATAL EXCEPTION`, or QNN loading errors observed in collected logcat.

## Phase 2 Progress Update (2026-08-14)

- Transformer weight inventory confirmed: 521 tensors, total 24,619,634,944 bytes; host RAM is 31.52 GiB with about 15.5 GiB available, so full Transformer loading/export is intentionally deferred to avoid another weight-loading stall.
- Existing VAE ONNX was converted successfully with QAIRT GPU backend after setting `QAIRT_TMP_DIR` to `E:\projects\zimage on phone\zimage-runtime\qairt-tmp`.
- VAE QNN artifacts: `build/qnn_vae_gpu/model.cpp`, `model.bin`, and `model_net.json`; Android library `build/android/jniLibs/arm64-v8a/libqnn_vae_gpu.so`, 199,053,976 bytes.
- VAE Android library was compiled with NDK 28.0.13004108 for `arm64-v8a` and synchronized to the connected `PJZ110` shared runtime. Remote size verified as 199,053,976 bytes.
- Incremental deployment script now handles missing remote files without fatal errors and all modified PowerShell scripts parse successfully.
- The application continues to use the existing dynamic QNN backend status path; actual VAE execution is not yet wired into JNI.

## VAE Runtime Load Investigation (2026-08-14)

- JNI now reports `VAE GPU graph` separately from the three QNN backend libraries.
- Device test exposed a real loader issue: `libqnn_vae_gpu.so` is present, but `dlopen` reports unresolved `_binary_obj_binary_*_raw_start` symbols.
- Root cause: the Windows NDK template invokes `llvm-objcopy` with an absolute `E:\qnn_android_build\...` input path, producing `_binary_E__qnn_android_build_*` symbols while generated QNN C++ expects `_binary_obj_binary_*`.
- Several temporary rebuild attempts confirmed the QNN C++ compiles, but the stock make rule still overrides the relative-path workaround. The VAE library on the device must therefore be treated as not loadable until rebuilt with a corrected template or Linux/WSL toolchain.
- Existing System/GPU/HTP backend loading remains verified and unaffected.

## VAE Loader Fix Completed (2026-08-14)

- Fixed Windows NDK QNN model-library embedding: weight objects are generated with relative `obj/binary/*.raw` paths, producing the expected `_binary_obj_binary_*` symbols instead of absolute-path symbols.
- Added QAIRT Linux platform helper source to the temporary Android model build and exported `strnDup` with default ELF visibility; the final library now resolves generated model and wrapper symbols.
- Final VAE library: `build/android/jniLibs/arm64-v8a/libqnn_vae_gpu.so`, 199,008,400 bytes.
- Deployed to `PJZ110` shared runtime and forced private staging refresh.
- Device UI verification passed: `System: loaded`, `GPU: loaded`, `HTP: loaded`, `VAE GPU graph: loaded`. No crash or fatal exception observed.
- This confirms VAE graph dynamic loading only; actual QNN graph execution and image output are still pending JNI graph execution integration.

## VAE QNN JNI Integration Update (2026-08-14)

- Implemented dynamic QNN GPU interface loading in android/app/src/main/cpp/zimage_runtime.cpp.
- Added QNN backend/device/context initialization, generated model symbol lookup, QnnModel_composeGraphs, graph retrieval, tensor metadata reporting, and a zero-latent VAE graphExecute probe with output min/max.
- Added QAIRT include path to android/app/src/main/cpp/CMakeLists.txt.
- Moved runtime initialization off the Android main thread in MainActivity.kt; the UI now renders first and disables the probe button until initialization returns.
- Added project-local debug keystore configuration because the host Android toolchain attempted to write its default keystore under inaccessible C:.android.
- APK build succeeded with Gradle 8.11.1, JDK 17, Android SDK/NDK 28.0.13004108.
- Device 3B15B100YVR00000 accepted the APK after uninstalling the previous differently-signed debug build. Runtime deployment completed to shared storage.
- Device process remained stable with no native crash. Observed app memory after startup: approximately 60 MB PSS / 5.9 MB native heap, indicating no large VAE execution allocation yet.
- Final UI probe result remains pending because the device returned to the lock screen during automated button verification; unlock and tap Generate test image to capture the graph execute result.

## Transformer ONNX Export Progress (2026-08-15)

### Environment repair
- `python312` runtime numpy was corrupted during migration (cp311 binaries mixed into cp312). Fixed with `pip install --force-reinstall numpy==2.5.2`.
- The Hermes agent's `PYTHONPATH` (`C:\Users\Max\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages`) leaks into the QAIRT Python 3.12 runtime and breaks imports (PIL `_imaging` conflict). Always run QAIRT python with `PYTHONPATH=` cleared or set to QAIRT's own `lib\python`.
- Confirmed working toolchain: torch 2.7.0+cpu / transformers 4.51.3 / diffusers 0.39.0 / numpy 2.5.2 in `zimage-runtime\python312`.

### Transformer call convention (critical for export)
- `ZImageTransformer2DModel.forward(x, t, cap_feats, ...)` expects **lists** in standard mode: `x` is a list of `(C, F, H, W)` tensors (latent `[1,16,64,64]` → `unsqueeze(2)` → `unbind(0)` → `[(16,1,64,64)]`), `cap_feats` is a list of **2D** `(seq_len, 2560)` tensors (no batch dim — pipeline masks out padding via `prompt_masks`).
- Output: list of `(C, F, H, W)`; wrapper stacks and squeezes to `[1,16,64,64]`.
- Model verified with dummy inputs; 6.15B parameters, output `[16,1,64,64]`.

### Complex64 RoPE patch (required for ONNX export)
- Problem: TorchScript ONNX exporter fails with `ScalarType ComplexFloat is an unexpected tensor scalar type` because Z-Image RoPE uses `torch.polar` + `view_as_complex` (complex64).
- Fix: patched `python312/Lib/site-packages/diffusers/models/transformers/transformer_z_image.py` (backup: `transformer_z_image.py.bak_complex`):
  - `precompute_freqs_cis`: emits interleaved `[cos, sin]` float32 pairs per frequency (identical layout to complex64 storage) instead of `torch.polar`.
  - `apply_rotary_emb`: real-valued rotation `(a*cos - b*sin, a*sin + b*cos)` instead of complex multiply.
- Verified numerically identical: max abs diff ~4.8e-07 vs original complex implementation.

### pad_sequence ONNX issue
- `torch.nn.utils.rnn.pad_sequence` is not ONNX-traceable (`aten::pad_sequence` unsupported). `export_transformer_onnx.py` monkey-patches the module-level binding inside the transformer with a static cat/stack equivalent.

### Export results
- Script: `tools/export_transformer_onnx.py` (add `--dtype fp32|fp16`, default fp16; opset 17).
- FP32 export: `build/transformer.onnx` + external weight files (~24.6 GB total). Verified with ONNX Runtime CPU: max abs diff 1.07e-04 vs PyTorch (rel 2.3e-05) — **PASS**.
- FP16 export: `build/transformer_fp16.onnx` — **DONE** (completed 2026-08-15 ~23:08; main graph 1.8 MB + external weights ~12.3 GB expected; verification run in background at write time).
- Note: FP32 was exported first as numerical baseline; FP16 is the intended GPU target per Phase 2 plan (Adreno FP16 → INT4 later).

### QAIRT conversion script (ready)
- `tools/convert_transformer_gpu.ps1`: converts `build/transformer_fp16.onnx` → `build/qnn_transformer_gpu/transformer.cpp|.bin` with fixed input dims (`latent 1,16,64,64`, `timestep 1`, `cap_feats 512,2560`) and `--no_simplification` (the proven workaround from text-encoder segments). GPU backend selected per VAE's successful configuration.
- Expected output: cpp ~2-3 MB, bin ~12 GB (FP16) — large; confirm free space before running (E: had 247 GB free).

### Device status (as of 2026-08-15)
- Device `3B15B100YVR00000` (PJZ110) is **connected**.
- APK 0.1 installed; QAIRT full library set + 6 segment libs + `libqnn_vae_gpu.so` already deployed at `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/lib/arm64-v8a/`.
- Models NOT deployed to device yet (large; deploy on demand with `deploy_android_runtime.ps1 -IncludeModel`).

### Next steps
1. Wait for FP16 export; run `tools/convert_transformer_gpu.ps1`.
2. Build Android model library (`libqnn_transformer.so`) following `tools/build_qnn_android_libs.ps1` pattern.
3. Extend JNI `zimage_runtime.cpp`: compose Transformer graph, wire text-encoder segments → Transformer → VAE with 8-step scheduler loop.
4. Deploy and verify on device.

---

## 2026-08-18 evening: GPU backend 打通与 FP16 溢出定位

- 调研结论：libQnnGpu.so 与 HTP 同 API；设备 /vendor/etc/public.libraries.txt 有 libOpenCL.so；
  APK 必须 `<uses-native-library libOpenCL.so>` 且不得打包 vendor OpenCL（否则 clGetPlatformIDs=-1001）。
- 阶段 0 shell CLI：qnn-net-run GPU 跑 VAE 成功但随机 latent 输出全 NaN（HTP 同样输入正常）。
  --debug 定位 VAE 第一个 NaN 在 attention Softmax（前级 MatMul 在 GPU 全 inf、HTP 全 NaN，HTP Softmax 把 NaN 变成 0 所以幸存）。
- v6 frontend GPU 被 `Convert FP32 cap_mask→FP16` 拒绝；v7 改 cap_mask FP16 + unified_mask FP16 后 GPU/HTP 均 compose+execute。
- v7 输出仍 NaN/Inf：--debug 首个非有限 `_noise_refiner_0_attention_to_out_0_MatMul_output_0_fc`（FP16 FC 累加溢出）。
- 同 FP16 ONNX 在 PC onnxruntime CPU 输出完全有限（unified min -61.4 max 740），确认是 QNN FP16 MatMul/FC 累加实现问题。
- 已构建：v7 FP16 frontend .so、v9 FP32 frontend .so（2.8GB）、v9 probe 输入与 ORT 参考 raw、
  tools/test_qnn_gpu_frontend.ps1、tools/convert_transformer_v9_fp32_all.ps1。
- 已改 APK：Manifest libOpenCL；移除 APK vendor OpenCL/dependency shims；build.gradle packaging excludes；
  C++ backendKind=2 GPU 路径 + OpenCL probe；APK assembleDebug 成功（36MB）。
- deploy_android_runtime.ps1 默认改为 v6 FP16 + text FP16。
- 设备在 v7 HTP/GPU 对照测试中途断开；以上全部设备验证待重连后执行。

---

## 2026-08-18 night: QNN GPU + v10 FP32 Transformer 全链在 app 内跑通

- 设备重连后 v9 FP32 frontend GPU/HTP 复测：GPU 无 NaN 且与 ORT 前端参考 maxabs=0.26、corr=0.99999998；HTP 对 FP32 仍输出 NaN（HTP MatMul/FC 数值路径仍溢）。
- 全量 v9 6-layer FP32 段（4.34GB model.bin）构建失败：aarch64 链接器 ADR_PREL_PG_HI21 越界。改 v10：export `--layers-per-segment 3`，10 个 layer 段（每段 2.17GB）+ frontend 2.92GB + final。
- qnn-net-run GPU 全链手工接续（frontend → 10 layer → final）成功：noise_pred vs ORT 全链参考 corr=0.999996、maxabs=0.019。
- APK/C++ 改为 dtype 自适应 probe（FP16→5×6-layer，FP32→10×3-layer），并修复 frontend tensor 指针 use-after-free（导致 app 内 layer mask/adaln 全 0、noise corr 仅 0.57）。
- app 内 GPU 复测通过：OpenCL `clGetPlatformIDs rc=0 num=1`，`GPU backendCreate/deviceCreate/contextCreate=0`；v10 FP32 12 段全部执行，noise_pred 与 CLI 完全一致、与 ORT corr=0.999996。
- app GPU 全链各段 execute 时间：frontend 818ms、10 层各约 1.7-2.0s、final 21ms（约 21s 纯执行；每段 prepare/finalize 另计）。
- 待修：GPU 上 VAE FP16 probe 输出全 NaN（HTP 正常）。

---

## 2026-08-19 night: HTP “graphFinalize=6001 / 被 v4 flash 搞坏” 真相定位与修复

### 结论（一句话）
设备 NPU 和 QNN HTP 一直是好的；app 里 100% `graphFinalize=6001` 的根因是
`android/app/src/main/cpp/zimage_runtime.cpp` 中自定义 `GraphInfo` 结构体字段顺序与 QNN SDK
的 `share/QNN/converter/jni/QnnWrapperUtils.hpp` 不一致 —— 代码把 `graphName` 字符串指针
当成了 `Qnn_GraphHandle_t` 传给 `graphFinalize`。

### 证据链
1. 设备侧 CLI 全程正常：同会话 `qnn-net-run` + `libQnnHtp.so` 的 VAE compose/finalize/execute 成功
   （`/data/local/tmp/htpdiag/out/Result_0/image_native.raw` 有限：min -0.5385742 / max 0.33007812）。
   `backendCreate/contextCreate=0`，且所有 QNN lib md5 与 SDK 一致 —— 排除库损坏。
2. 但 app 里 VAE 和 embedding 都在 compose 后 `graphFinalize=6001`，且单/双 backend 模式、甚至
   删掉 `libQnnGpu.so` 都复现 —— 排除 flash/库/双 backend。
3. 加入 `VAE graphPtr=... name=... finalizeFn=...` 日志后，app 在打印 `name=` 时
   `SIGSEGV fault addr 0x1`（`strlen(0x1)`，backtrace `composeVae+1752`，addr2line 指到
   `operator<<(char const*)` 读 `graphName`）。说明 `graphName` 读到的是 0x1 —— 这正是一个合法的
   QNN graph handle（graph 1），而不是字符串指针。
4. 对照 SDK 头：官方 `GraphInfo` 是
   `{Qnn_GraphHandle_t graph; char *graphName; Qnn_Tensor_t *inputTensors; uint32_t numInputTensors; ...}`；
   当时 cpp 里写成了 `{graphName, graph, numInputTensors, inputTensors, ...}`。因此：
   - `r.graphs[0]->graph` 实际读到偏移 8 的 `graphName`（非空堆指针）→ app 以为 graph 已存在，
     跳过 `graphRetrieve`；
   - `graphFinalize(该 graphName 指针)` → QnnGraphTransformerAdapter 拿到非法 handle → 6001。
   - CLI 用 SDK 自己的正确结构体，所以永远正常。
5. 顺带发现旧结论“QNN 单进程不能双 backend 共存（6001）”也是在错误结构体下得出的，
   不能作为 backend 冲突证据；双 backend 待用正确结构体复测。

### 修复与验证
- `GraphInfo` 恢复为官方字段顺序（`Qnn_GraphHandle_t graph; char* graphName; Qnn_Tensor_t* inputTensors;
  uint32_t numInputTensors; Qnn_Tensor_t* outputTensors; uint32_t numOutputTensors;`）。
- `composeSegment` 的 finalize/graphRetrieve 统一用传入 QnnSet 的 `s.api`（不再用全局 `r.api`）。
- Gradle 重新 `assembleDebug`（`android/app/build/outputs/apk/debug/app-debug.apk`），安装后
  `backend.txt=htp` 实测：
  - `VAE graphPtr=1 name=vae finalizeFn=...` → `graphFinalize=0`
  - VAE probe 输出有限：min=-0.538574 max=0.330078 mean=0.00953321
  - text encoder embedding + 6 个 layer 段全部 compose/finalize/execute 成功，
    `final_hidden[-2]` min=-702.5 max=2506 mean=0.202791（与既有 FP16 数值闭环一致）
  - jni.log 中 `graphFinalize=6001` 计数 = 0。
- 设备没有重启、没有刷机；问题纯软件，修复成本 = 一行结构体顺序。

### 教训
- 任何手写 QNN wrapper ABI 结构体必须逐字段对照 SDK 头（`QnnWrapperUtils.hpp`），字段顺序一个
  都不能猜；这类错误不会在 compose 阶段报错，而是在 finalize/execute 阶段变成诡异的 6001 或 1001。
- 修 debug 日志时不要直接解引用 model 返回的指针；本次抓到的 `strlen(0x1)` crash 反而是定位关键。
