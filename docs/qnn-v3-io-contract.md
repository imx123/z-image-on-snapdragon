# QNN v3 float-IO 契约（Transformer / Qwen3 / VAE）

生成时间：2026-08-16 evening。所有信息来自 `build/qnn_manifest_v3/*.json`
（由 `tools/generate_qnn_manifest.py` 从 `model_net.json` 生成）。

## 0. 为什么是 v3

- v2 分段 IO 是 `UFIXED_POINT_8`，且每段校准独立随机，段边界编码严重失配。
- v3 用 `--preserve_io datatype`：图 IO 保持 ONNX 原始 dtype
  （Transformer latent/timestep/cap/unified/adalm 为 FP16，freqs/mask 为 FP32；
  Qwen3 全部 FP32；VAE FP16），内部权重/激活仍是 W4A8 / W8A8。
- 因此 JNI 边界**不需要** CPU 量化/反量化，只需要做 layout transpose。

## 1. Transformer v3（W4A8，内部 u8 + 4-bit weights）

模型目录：`build/qnn_transformer_segments_v3_floatio_w4a8/`
Android 库：`build/android/jniLibs-transformer-v3/arm64-v8a/libqnn_transformer_<seg>.so`

| 段 | 库 | 输入（QNN dims / dtype） | 输出（QNN dims / dtype） |
|---|---|---|---|
| frontend | libqnn_transformer_frontend.so | latent `[1,64,64,16]` fp16（src `[1,16,64,64]`, perm 0,3,1,2）；timestep `[1]` fp16；cap_feats `[512,2560]` fp16；cap_mask `[512]` bool8 | unified `[1,1536,3840]` fp16；unified_mask `[1,1536]` fp32；adaln_input `[1,256]` fp16。**没有 unified_freqs 输出**（converter 常量折叠掉了） |
| layers_XX_YY | libqnn_transformer_layers_XX_YY.so | unified_in `[1,3840,1536]` fp16（src `[1,1536,3840]`）；unified_freqs `[1,128,1536]` fp32；unified_mask `[1,1536]` fp32；adaln_input `[1,256]` fp16 | unified_out `[1,3840,1536]` fp16 |
| final | libqnn_transformer_final.so | unified_in `[1,3840,1536]` fp16；adaln_input `[1,256]` fp16 | noise_pred `[1,16,64,64]` fp16 |

**unified_freqs 常量资产**（必须部署到 `<runtime>/assets/`）：
`build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw`
- QNN dims `[1,128,1536]` float32，由 ONNX 逻辑 `[1,1536,128]` transpose (0,2,1) 得到；
- 已验证与 latent/timestep/cap_feats/cap_mask 无关。

**scheduler 约定**：送入 scheduler 前 `noise_pred = -noise_pred`；v3 probe 暂不做取反，
便于与 ONNX chain 数值对比。

**对比基准**（随机 calib sample_00）：
`E:\zimage_calib_transformer_chain\final\sample_00_noise_pred.raw`
min=-6.8203125, max=7.203125, mean=-0.0591515, std=1.205988。

## 2. Qwen3 v3（W8A8 内部，FP32 IO）

模型目录：`build/qnn_text_encoder_segments_int8_floatio/`（layers_XX_YY）
以及 `build/qnn_text_encoder_embedding_fp16/`（embedding，FP16 weights）
Android 库：`build/android/jniLibs-text-int8-floatio/arm64-v8a/`

| 段 | 库 | 输入（QNN dims / dtype） | 输出 |
|---|---|---|---|
| embedding | libqnn_embedding.so | input_ids `[1,512]` int64 | hidden_states `[1,512,2560]` fp32（逻辑布局） |
| layers_XX_YY | libqnn_layers_XX_YY.so | hidden_states `[1,2560,512]` fp32（src `[1,512,2560]` perm 0,2,1）；attention_mask `[1,512,512,1]` fp32；cos `[1,128,512]` fp32；sin `[1,128,512]` fp32 | hidden_states_out `[1,2560,512]` fp32 |

预计算资产（必须部署到 `<runtime>/assets/`）：
- `cos_qnn_f32.raw`、`sin_qnn_f32.raw`：`[1,128,512]` fp32，逻辑 `[1,512,128]` 的 transpose。
- `causal_mask_qnn_f32.raw`：`[1,512,512,1]` fp32，下三角 `-inf`、上三角 0；
  runtime 需按 `attention_mask[k]==0` 再把整列置 `-inf`。

**重要**：最后一层段输出就是官方 pipeline 使用的 `hidden_states[-2]`
（layer 35 输出、final RMSNorm 之前）。不要接 `final_norm.onnx`。

## 3. VAE（FP16）

- 模型：`build/qnn_vae_shiftfix_fp16/vae.cpp/.bin`
- Android：`build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so`
- IO：latent `[1,64,64,16]` fp16 → image `[1,512,512,3]` fp16。
- ONNX 内部已包含 `/scaling_factor + shift_factor`（0.3611 / 0.1159）。

## 4. JNI probe 调用

- `ZImageRuntime.transformerProbe(latent: FloatArray(65536), timestep: Float, capFeats: FloatArray(1310720), capMask: BooleanArray(512))`
- `ZImageRuntime.textEncoderProbe(inputIds: IntArray(512), attentionMask: BooleanArray(512))`
- 两个 probe 都先 `ensure HTP context`，然后串行 compose → execute → free
  （text encoder 每次只常驻一段，峰值内存低）。
