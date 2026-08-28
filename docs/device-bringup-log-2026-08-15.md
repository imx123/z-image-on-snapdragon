# Device bring-up log — 2026-08-15 (PJZ110, OnePlus 13)

Session log for the first real-device install/test pass. Device: `PJZ110`
(OnePlus 13, Snapdragon 8 Elite), arm64-v8a, Android 16 (ColorOS), adb serial
`3B15B100YVR00000`.

## 1. Install & deploy (done)

- `adb install -r app-debug.apk` → Success. APK is the lightweight debug build
  (~5.6 MB); all multi-GB QNN artifacts live under
  `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime`
  (shared storage), staged into app-private storage at startup.
- Runtime libraries verified on device before launch:
  - 6 × `libqnn_layers_*.so` (each 2,423,451,256 bytes) — text encoder segments
  - `libqnn_vae_gpu.so` (199,008,400 bytes)
  - Full QAIRT 2.49 aarch64-android lib set (103 files total)
- App launches, renders UI, no crash. `adb shell pidof` stable.

## 2. Bug 1 — staging copy silently failed: shared-storage libs unreadable

**Symptom**: UI showed `System: missing / GPU: missing / HTP: missing /
VAE library: missing / VAE graph: GPU interface unavailable` even though the
libs were present under shared storage.

**Root cause**: files pushed via `adb` are owned by `shell:ext_data_rw` with
mode `770`. The app runs as `u0_a553` and is neither owner nor in the group,
so `MainActivity.stageRuntimeFiles()` saw `sourceLibs.isDirectory() == false`
and skipped the copy entirely. Private dir `/data/user/0/.../files/zimage-runtime`
never got created.

**Fix (device-side, no rebuild needed)**:
```
adb shell chmod -R a+rX /storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/
```
After the chmod, staging copied all 103 files (~14 GB, took ~1.5 min) and the
UI reported:
```
System: loaded
GPU: loaded
HTP: loaded
VAE library: loaded
VAE graph: backendCreate=1006
```
Note: `deploy_android_runtime.ps1` does not fix ownership/mode after `adb push`.
Make the chmod part of the deploy script (or `adb shell chown` to the app uid).

## 3. Bug 2 — GPU backend create fails: error 1006

**Symptom**: `VAE graph: backendCreate=1006`.

**Decode**: 1006 == `QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED`
(QNN_MIN_ERROR_COMMON=1000 + 6, see `QnnCommon.h`).

**Root cause chain**:
1. `libQnnGpu.so` (readelf NEEDED: libEGL, libGLESv2, libc/m/dl/log only) loads
   OpenCL at runtime via `dlopen("libOpenCL.so")` + `clGetPlatformIDs`.
2. The device's `/vendor/lib64/libOpenCL.so` is a **KHR ICD loader**
   (symbols: `khrIcdInitialize`, `khrIcdOsVendorsEnumerate`, ...), not a real
   implementation. The real OpenCL is `/vendor/lib64/libOpenCL_adreno.so`
   (exports `clIcdGetPlatformIDsKHR`).
3. The loader finds vendor implementations only through `.icd` files under
   `/vendor/etc/OpenCL/vendors/` (or `OCL_ICD_VENDORS` env). **This device has
   no `/vendor/etc/OpenCL/vendors/` at all** (`find / -name '*.icd'` → empty).
4. So `clGetPlatformIDs` returns 0 platforms → libQnnGpu backendCreate fails
   with 1006.

## 4. Attempted fixes for 1006 (none succeeded yet)

### 4a. OCL_ICD_VENDORS env var + local .icd (JNI change)

Patched `zimage_runtime.cpp init()`:
- create `<runtime root>/ocl_vendors/adreno.icd` containing
  `libOpenCL_adreno.so`
- `setenv("OCL_ICD_VENDORS", ...)`, `setenv("OCL_ICD_FILENAMES", ...)`
- also pushed `libOpenCL_adreno.so` into shared storage so staging copies it

Result: still 1006. Diagnostics in logcat (`zimage-jni` tag) showed:
```
direct dlopen adreno: dlopen failed: library "libcutils.so" not found:
  needed by .../libOpenCL_adreno.so in namespace clns-9
dlopen libOpenCL.so: dlopen failed: library "libOpenCL.so" not found
  env=/data/user/0/.../ocl_vendors/adreno.icd
```
Two separate problems:
- `libOpenCL_adreno.so` needs vendor libs (`libcutils.so`, `libvndksupport.so`)
  which are NOT in the app's classloader namespace (clns-9).
- Even `dlopen("libOpenCL.so")` by bare name fails: the loader lives in
  `/vendor/lib64`, listed only in `/vendor/etc/public.libraries.txt`
  (system public.libraries.txt does not contain it). The app namespace cannot
  see it by name.

### 4b. Ship OpenCL stack inside APK jniLibs (in progress)

Copied 4 libs into `android/app/src/main/jniLibs/arm64-v8a/`:
- `libOpenCL.so` (the KHR loader)
- `libOpenCL_adreno.so` (the real implementation)
- `libcutils.so` (vendor dep, pulled from /system/lib64)
- `libvndksupport.so` (vendor dep, pulled from /system/lib64)

**BLOCKER discovered while verifying**: `app/build.gradle` overrides
`jniLibs.srcDirs` to point at `${rootDir}/../build/android/apk-jniLibs`
(kept intentionally light). Our new libs under `src/main/jniLibs/` were NOT
packaged — the APK still contains only `libzimage_runtime.so`.

Next step: either
- copy the 4 libs to `E:\projects\zimage on phone\zimage on phone\build\android\apk-jniLibs\arm64-v8a\`, or
- add a second jniLibs srcDir / drop the override for this bring-up.

Then rebuild, reinstall, and re-check `backendCreate`. Open question whether
`dlopen("libOpenCL.so")` inside libQnnGpu (called from our JNI which was loaded
from APK lib dir) resolves against APK-bundled libs — the earlier bare-name
failure was from a namespace that had no libOpenCL at all; bundling should make
it resolvable, but the KHR loader also consults its own search order for the
`.icd` target (`libOpenCL_adreno.so` as bare name, which would resolve inside
the same namespace if the loader itself was loaded from the APK).

### 4c. Fallback considered

If bundling the OpenCL stack does not work (namespace isolation, secure_getenv
blocking `OCL_ICD_VENDORS`, etc.), options:
- Try HTP backend for VAE instead of GPU (`libqnn_vae_gpu.so` is GPU-compiled;
  would need a re-export for HTP) — large effort, last resort.
- Investigate whether `libOpenCL_adreno.so` can be loaded with
  `android_dlopen_ext` + `ANDROID_NAMESPACE` tricks.
- Check for a QNN GPU backend env/config that bypasses OpenCL enumeration.

## 5. Environment notes (paths on this machine)

- Project root: `E:\projects\zimage on phone\zimage on phone`
- Runtime root: `E:\projects\zimage on phone\zimage-runtime`
- QAIRT: `E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730`
- NDK readelf: `C:\Users\Max\AppData\Local\Android\Sdk\ndk\28.0.13004108\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-readelf.exe`
- Gradle: `C:\Users\Max\.gradle\wrapper\dists\gradle-8.11.1-bin\...\gradle.bat`
- Build: `export GRADLE_USER_HOME=E:\projects\zimage on phone\zimage-runtime\gradle-home` etc. (see PROJECT_HANDOFF.md)

## 6. Current file state (uncommitted changes)

- `android/app/src/main/cpp/zimage_runtime.cpp`:
  - added includes `<cstdio>`, `<cstdlib>`, `<sys/stat.h>`
  - `init()` now writes `ocl_vendors/adreno.icd`, sets `OCL_ICD_VENDORS` /
    `OCL_ICD_FILENAMES`, and logs dlopen diagnostics under tag `zimage-jni`
- `android/app/src/main/jniLibs/arm64-v8a/` (NEW, not yet packaged):
  libOpenCL.so, libOpenCL_adreno.so, libcutils.so, libvndksupport.so
- No changes to MainActivity.kt / ZImageRuntime.kt yet.

## 7. Quantization audit (2026-08-15) — CRITICAL

**Finding**: the deployed text encoder is NOT quantized. The converter command
recorded `weights_bitwidth=8`, but `conversion.log` says:

```
Skipping quantization, no input_list provided
```

No calibration data (input_list) was supplied, so QAIRT skipped quantization
entirely and all 530 tensors per segment have `bitwidth=0` (float32). Result:

| Component | Params | Deployed format | Deployed size | Verdict |
|-----------|--------|-----------------|---------------|---------|
| Text encoder (Qwen3) | 4B | **float32** | **13.5 GiB** (6×2.26GiB) | 🔴 way over budget |
| VAE | 84M | float32 (opset18) | 0.19 GiB | 🟡 ok for now |
| Transformer | 6.15B | not deployed (fp32 24.6GB source) | — | ⏳ Phase 2 |

**Memory constraint**: device has 24 GB total / ~15 GB usable. Current text
encoder alone (13.5 GiB) blows the budget before VAE/runtime/OS.

**Chosen plan (方案 A, recommended)**:
- Text encoder → **int8** (~4 GB total, ~670 MB/segment) with real
  calibration input_list
- Transformer (Phase 2) → **int4** weight-only (~3.1 GB)
- VAE → keep fp16/fp32 (0.19 GB, not a bottleneck)
- Estimated peak ~7.5 GB, comfortable inside 15 GB

**Steps in progress**:
1. Generate calibration inputs for all 6 segments (extend
   `tools/prepare_text_inputs.py`)
2. Re-run `tools/convert_qwen3_segments.ps1` with
   `--input_list` + `--weights_bitwidth 8` + `--act_bitwidth 8`
3. Verify each segment drops from 2.26 GiB → ~670 MB
4. Redeploy to device, verify actual RSS

Calibration data approach: run the real tokenizer on a few representative
prompts (512 tokens each), export `hidden_states/attention_mask/cos/sin`
inputs for each segment as raw binary. QAIRT min-max calibration needs actual
activations, so per-segment calibration inputs should be generated by running
each segment's ONNX forward on the host with sample data (ORT), or by feeding
the segment input tensors directly.

## 8. Deployment notes (2026-08-15)

- Device shared-runtime chmod fix (see section 2) must be added to
  `tools/deploy_android_runtime.ps1` so future deploys don't regress.
- Rebuilt APK with OpenCL-stack jniLibs attempt; **not yet packaged** due to
  `build.gradle` `jniLibs.srcDirs` override (see section 4b). Next build must
  either copy the 4 libs into `build/android/apk-jniLibs/arm64-v8a/` or drop
  the srcDirs override.

## 9. int8 quantization landed (2026-08-15)

### Calibration data
- Script: `tools/generate_qwen3_calibration.py` — runs the real tokenizer +
  Qwen3 forward pass on 8 prompts (512 tokens), chains hidden states
  segment-by-segment, writes per-segment raw inputs.
- **fp16 model load** required: fp32 (16 GB weights) OOM-crashes this host
  (exit 139) during the chained forward. Calibration only needs activation
  ranges, so fp16 is fine; raw files are still written as float32.
- **input_list format gotcha**: QAIRT expects ONE line per calibration sample,
  with each graph input as a SEPARATE raw file, paths separated by spaces —
  NOT one concatenated blob per line. Also: **no spaces allowed in any path**
  (`zimage on phone` breaks parsing), so calibration files live at
  `E:\zimage_calib_v2` (no-space root).
- Output: `E:\zimage_calib_v2\layers_XX_YY\{sample_NN_hidden,mask,cos,sin}.raw`
  + `input_list.txt` (8 samples/segment, 6.8 MB/sample). Working copy also at
  `build/qwen3_calibration_v2/` (paths there contain spaces — use E:\ root
  copy for QAIRT).

### int8 conversion
- Script: `tools/convert_qwen3_segments_int8.ps1` (new) — adds
  `--input_list`, `--weights_bitwidth 8`, `--act_bitwidth 8`,
  `--bias_bitwidth 32`, `--param_quantizer tf`, `--act_quantizer tf` on top of
  the original fp32 conversion flags.
- **Result: all 6 segments converted successfully.**
  - `build/qnn_text_encoder_segments_int8/layers_*/model.bin` =
    605,798,400 bytes each (~578 MiB) vs 2,422,548,480 fp32 (~2.26 GiB) —
    **exactly 1/4 size, int8 as expected.**
  - Verified in `model_net.json`: 506 tensors bitwidth=8, 24 tensors
    bitwidth=32 (biases/constants), 530/530 have scale/offset encodings.
- Run with:
  ```
  $env:QNN_SDK_ROOT = 'E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730'
  $env:QAIRT_PYTHON  = 'E:\projects\zimage on phone\zimage-runtime\python312\python.exe'
  $env:QAIRT_TMP_DIR = 'E:\projects\zimage on phone\zimage-runtime\qairt-tmp'
  .\tools\convert_qwen3_segments_int8.ps1 -Force
  ```

### Android .so build — BLOCKED (in progress at pause)
- `tools/build_qnn_android_libs.ps1` was extended to extract `model.bin` with
  Python's tarfile (tar.exe treats `E:\...` as a remote host). Extraction now
  works.
- **ndk-build fails**: `QnnInterface.h not found` because `QNN_SDK_ROOT`
  contains spaces (`E:\projects\zimage on phone\...`) and ndk-build's include
  line splits it: `-IE:/projects/zimage -Ion -Iphone/zimage-runtime/...`.
  Same class of problem as the tar issue. The previous fp32 build succeeded
  only because QAIRT lived at the no-space `E:\zimage-runtime` path pre-migration.
- Fix candidates (next session):
  1. Copy `include/QNN` into `E:\qnn_android_build\<segment>\` and point
     `LOCAL_C_INCLUDES` at the local copy (no spaces) in the generated
     Android.mk.
  2. Use the 8.3 short path for QNN_SDK_ROOT (hard with spaces in middle dirs).
  3. Symlink/junction `E:\qnn` → `E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730` and use `E:\qnn` in the mk.

### Memory budget (post-int8, projected)
- Text encoder int8: 6 × 0.578 GiB ≈ **3.5 GiB** (was 13.5 GiB fp32) ✅
- VAE fp16/fp32: 0.19 GiB ✅
- Transformer (Phase 2): int4 target ≈ 3.1 GiB
- QAIRT/QNN runtime libs: ~0.5 GiB
- **Estimated peak ≈ 7.5 GiB — comfortably inside 15 GB usable.**



## 8. 2026-08-16 breakthrough — HTP backend works, GPU/OpenCL is a dead end (in sandbox)

### Verification (on-device, via jni.log file logging)
- Added file logging to JNI (writes to <runtime root>/jni.log, read via run-as).
- Full dependency chain for bundled OpenCL now resolves:
  - libOpenCL.so / libOpenCL_adreno.so / libc++.so / libbase.so / libcutils.so
    + custom stub libvndksupport.so (SONAME + LIBVNDKSUPPORT versioned export of
    android_load_sphal_library) all dlopen OK.
- **BUT** clGetPlatformIDs → rc=-1001 (CL_INVALID_PLATFORM), and direct
  adreno clIcdGetPlatformIDsKHR → rc=-30.
- Root cause: libOpenCL_adreno.so internally dlopens /vendor libs
  (libCB.so, libgsl.so, libq3dtools_adreno.so, searches /system/vendor/lib64/egl)
  which are NOT accessible from the app's clns-9 namespace. Also may need
  /dev/kgsl-3d0. **OpenCL in a non-root app sandbox is not viable on this device.**
- **HTP probe: backendCreate=0 (SUCCESS)** — NPU path works in the app sandbox!

### Decision
- VAE: reconvert with backend=HTP (tools/convert_vae_htp.ps1, weights/act bitwidth 8).
- Text encoder segments: already HTP — good.
- Transformer: target HTP too (Phase 2 will convert fp16 ONNX with backend=htp),
  GPU/OpenCL only if a root/privileged path is ever available.

### APK packaging learnings (build.gradle jniLibs)
- jniLibs.srcDirs = QNN_ANDROID_JNILIBS env ?: \/../build/android/apk-jniLibs
- The 4 bundled OpenCL libs MUST be copied to build/android/apk-jniLibs/arm64-v8a/,
  NOT src/main/jniLibs (that dir is ignored by the srcDirs override).
- gradle 8.11.1 cache corruption: delete ~/.gradle/caches/8.11.1 to fix
  'Could not read workspace metadata' / groovy-dsl metadata errors.

## 9. 2026-08-16 (continued) — HTP device/context creation debugging

### QNN 2.49 interface signatures (critical, from QnnInterface.h)
- `backendCreate(Qnn_LogHandle_t logger, const QnnBackend_Config_t** cfg, Qnn_BackendHandle_t*)` — first arg is LOGGER (nullptr OK).
- `deviceCreate(Qnn_LogHandle_t logger, const QnnDevice_Config_t** cfg, Qnn_DeviceHandle_t*)` — first arg is LOGGER, NOT backend! Passing backend → 11003 INVALID_ARGUMENT.
- `contextCreate(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, cfg, Qnn_ContextHandle_t*)` — args are backend/device as expected.
- Single-NSP deviceCreate (official Genie QnnApi.cpp): pass config array `{nullptr}`; socModel/arch custom configs are only for multi-NSP.

### HTP creation sequence after fixes
```
HTP logCreate=0 handle=set          (QNN internal: QnnLog_create OK)
HTP backendCreate=0                 (QnnBackend_create done successfully)
deviceCreate=11003 → fixed logger arg → QnnDevice_create started
[QNN] Failed to create transport instance: 4000
[QNN] Failed to create transport for device, error: 4000
[QNN] Failed to load skel, error: 4000
[QNN] Transport layer setup failed: 14001
contextCreate=14001
```
- 4000 = QNN_BACKEND_ERROR_NOT_SUPPORTED; 14001 = QAIRT_DEVICE_ERROR_INVALID_CONFIG.
- Root cause: HTP needs fastRPC transport → libcdsprpc.so + write access to
  /dev/fastrpc-cdsp (crw-rw-r-- system system, SELinux vendor_qdsp_device).
  App `echo test > /dev/fastrpc-cdsp` → Permission denied.

### Fixes attempted
1. AndroidManifest: `<uses-native-library android:name="libcdsprpc.so" required=false>`
   — must be INSIDE <application> (AAPT error otherwise). Build OK, transport still failed.
2. Copied /vendor/lib64/libcdsprpc.so + libadsprpc.so (497 KB each) into staging
   files/zimage-runtime/lib/arm64-v8a/ via run-as cp. **Untested at pause** — next
   action is force-stop + relaunch + read jni.log.

### Fallback plan if fastRPC stays blocked
- Use CPU backend (libQnnCpu.so already staged; no fastRPC/OpenCL needed) to validate
  the full composeGraphs → graphExecute → VAE decode chain.
- HTP on-device may require root (chmod 666 /dev/fastrpc-cdsp or SELinux permissive)
  or platform-signed app.
