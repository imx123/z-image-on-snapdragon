# QNN text encoder target

The local ModelScope snapshot contains a 4B-parameter Qwen3 text encoder. The Android target uses a fixed `batch=1`, `sequence_length=512`, and returns `last_hidden_state` with shape `[1,512,2560]`.

Recommended conversion order:

1. Generate calibration inputs with `prepare_text_inputs.py`.
2. Export or partition the Qwen3 graph with KV cache disabled.
3. Quantize Linear/GEMM weights to INT8 for QNN HTP; keep RMSNorm, RoPE, and residual accumulation in FP16 where required.
4. Compile with Qualcomm QNN HTP only after checking operator coverage.
5. Cache the resulting `[1,512,2560]` conditioning tensor per prompt.

The repository does not download or bundle Qualcomm SDK binaries. Set `QNN_SDK_ROOT` externally when the SDK is available.
