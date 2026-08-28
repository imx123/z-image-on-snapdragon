# Text encoder conversion status

The local ModelScope Qwen3 text encoder has been exported at fixed shape `[1,512] -> [1,512,2560]` and passes QAIRT 2.49 dry-run after folding the constant `LessOrEqual` mask node.

QAIRT formal conversion currently fails in its forced `MATMUL_TO_FC`/`ALIGN_MATMUL_RANKS` optimization while reading the large external weight set. The converter's ONNX entry hard-codes `align_matmul_ranks=True`; this is a QAIRT 2.49 limitation for this graph, not an unsupported Qwen3 operator.

The next implementation path is segmented export: keep tokenizer and prompt processing on CPU, split Qwen3 into fixed layer groups, and compile each group as a smaller QNN graph. Conditioning output remains `[1,512,2560]`. This reduces conversion peak memory and makes Android-side stage loading possible.

The conversion helper was corrected for QAIRT 2.49: `qnn-onnx-converter` does not accept `--target_backend`, so backend selection belongs to the later model-library/context generation step. The helper now uses the existing `build/text_encoder_qnn.onnx` artifact, enables debug logging, and fails if QAIRT exits without producing `text_encoder.cpp`. A full-graph retry reached `Defer loading` but did not produce output within the verification window; the process was stopped. Continue with segmented layer-group export rather than retrying the full graph.

The segmented exporter is `tools/export_qwen3_segments.py`. Run it from an environment containing PyTorch and Transformers 4.51+:

```powershell
python tools/export_qwen3_segments.py `
  --model-root tools/models/z-image-turbo/models/Tongyi-MAI--Z-Image-Turbo/snapshots/master `
  --out-dir build/text_encoder_segments `
  --sequence-length 512 `
  --layers-per-segment 6 `
  --opset 17
```

The output contains isolated `embedding/model.onnx`, six `layers_XX_YY/model.onnx` graphs covering layers 0-35, `final_norm/model.onnx`, and `manifest.json`. Each layer graph's effective inputs are hidden states, additive mask, and fixed RoPE `cos`/`sin`; Transformers accepts `position_ids` during tracing, but ONNX removes it after the explicit RoPE tensors are supplied. Each graph directory owns its external weight files so same-named tensors cannot overwrite another layer group. Validate the file set and contiguous layer coverage with `tools/verify_qwen3_segments.py build/text_encoder_segments/manifest.json`.

The runtime Python environment at `E:\zimage-runtime\python312` now has CPU `torch 2.7.0`, `transformers 4.51.3`, and `safetensors 0.8.0`, installed from the Aliyun PyPI mirror. Verified artifacts are in `build/text_encoder_segments_v2/`: the manifest covers all 36 layers, every layer group has 54 external tensors with zero missing files, and ONNX Runtime executes `layers_00_05/model.onnx` with output shape `[1,512,2560]`.

`tools/monitor_qnn_conversion.ps1` was added for QAIRT conversion diagnostics. A monitored retry of `layers_00_05/model.onnx` showed stable private memory around 858 MB and working set around 115 MB while CPU time continued increasing, with no output or temp-file growth. This indicates a QAIRT 2.49 optimizer stall in the MatMul path rather than host memory exhaustion; the process was stopped after the timeout.

The decisive workaround is `--no_simplification`. QAIRT debug output showed the default ONNX path entering simplification and enabling `AlignMatmulRanks`; with `--no_simplification`, the same `layers_00_05/model.onnx` converted successfully to `qnn_layers_no_simplify.cpp` and `qnn_layers_no_simplify.bin`. The converter log ended with `Conversion complete!`.

Use `tools/convert_qwen3_segments.ps1` to convert all six layer groups. It writes one `model.cpp`/`model.bin` pair per group under `build/qnn_text_encoder_segments/`, skips complete existing pairs, and always passes `--no_simplification`.

All six groups have now converted successfully. Each group produced a `model.cpp` of about 1.37 MB and a `model.bin` of 2,422,548,480 bytes; logs end with `Conversion complete!`. The next QAIRT step, `qnn-model-lib-generator`, successfully extracts the BIN but cannot compile the Windows target because CMake reports `No CMAKE_C_COMPILER` and `No CMAKE_CXX_COMPILER` for the requested `ClangCL` toolset. The installed generator exposes Windows targets only; Android NDK/clang is not currently available in the host environment.

Environment recheck: Android Studio is installed at `C:\Program Files\Android\Android Studio`. The confirmed Android SDK is `C:\Users\Max\AppData\Local\Android\Sdk`, with NDK `28.0.13004108` at `C:\Users\Max\AppData\Local\Android\Sdk\ndk\28.0.13004108`. The SDK directory is protected from ordinary workspace enumeration, but `ndk-build.cmd`, clang, llvm-objcopy, and the Android toolchain were successfully invoked with controlled permissions.

Android QNN model libraries are now built with `tools/build_qnn_android_libs.ps1`. The script uses an external no-space temporary path (`E:\qnn_android_build`) because Windows NDK rejects project paths containing spaces, extracts each QAIRT BIN in PowerShell, and uses the Windows NDK `llvm-objcopy` instead of the template's Linux path. All six `arm64-v8a` libraries are present under `build/android/jniLibs/arm64-v8a/`: each is 2,423,451,256 bytes and contains 114 embedded raw weight objects. The generated libraries are ready for Android packaging; device execution still requires the matching QNN runtime/backend libraries and a real-device validation pass.

The Android APK intentionally does not package these six multi-GB libraries because APK/ZIP32 cannot represent the combined payload above 4 GiB. `android/app/build.gradle` uses the lightweight `build/android/apk-jniLibs` directory for APK JNI assets, while `tools/deploy_android_runtime.ps1` pushes the model libraries from `build/android/jniLibs` to the device separately. With `android/local.properties` pointing to the confirmed SDK, `assembleDebug` completes successfully.
