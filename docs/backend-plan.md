# Z-Image-Turbo 组件后端可行实现方案

日期：2026-08-16（night）
设备：PJZ110 / OnePlus 13 / Snapdragon 8 Elite (SM8750, Adreno 830 + Hexagon HTP v79)
QAIRT：2.49.0.260730

## 0. 结论速览

1. **不要再投入 QNN GPU / OpenCL 路径**：app 沙箱内 `backendCreate=1006`、`clGetPlatformIDs=-1001`，已证实不可行。
2. 当前设备上已实证可用的两个加速通道：
   - **QNN HTP / NPU**：VAE 已完成 compose → finalize → execute，`deviceCreate=0/contextCreate=0/graphExecute=0`。
   - **LiteRT GPU delegate（GLES/EGL）**：10x10 探针执行成功，`delegate=true`。
3. 建议的最终组合：
   - Tokenizer / 文本预处理：**CPU**
   - Qwen3 文本编码器：**QNN HTP，INT8，6 段**
   - Z-Image Transformer（6.15B DiT）：**QNN HTP，FP16 或权重 INT4**
   - FlowMatchEuler 调度器：**CPU**
   - VAE Decoder：**QNN HTP，FP16**
4. LiteRT GPU delegate 留作 Transformer/VAE 的 Plan B，但需要先把组件转成 TFLite/LiteRT，转换工作量和算子覆盖风险高于继续走 QNN HTP。


---

## 1. 各组件分析

### 1.1 Tokenizer / 提示词预处理 — CPU

- 模型：`Qwen2Tokenizer`，vocab 151936，BPE merges。
- 流程：`apply_chat_template(enable_thinking=True, add_generation_prompt=True)` → `padding=max_length(512)` → `truncation`。
- 输出：`input_ids[1,512]` 和 `attention_mask[1,512]`。
- 后端：**CPU**，资源 < 10 MB，几毫秒级开销。
- 现状：`build/text-inputs.json` 已有测试输入；运行时 tokenizer 尚未接入。

### 1.2 Qwen3 文本编码器 — QNN HTP INT8（分段）

- 结构：36 层，hidden=2560，32 Q heads / 8 KV heads，head_dim=128，intermediate=9728（GQA + SiLU MLP + RMSNorm + RoPE）。
- 参数量：**4,022,468,096（约 4.0B）**。
- 尺寸：
  - 原始 safetensors：约 8.0 GB（BF16）。
  - ONNX 分段 v2：约 18 GB（FP32）。
  - QNN INT8 六段：每段 `model.bin` 605,798,400 B（约 578 MiB），六段合计 **约 3.40 GB**。
- I/O（固定 seq=512）：
  - embedding：`input_ids[1,512]` → `hidden[1,512,2560]`
  - 每段：`hidden[1,512,2560]` + `attention_mask[1,1,512,512]` + `cos/sin[1,512,128]` → `hidden`
  - 给 Transformer 的 conditioning：`cap_feats[L,2560]`，`L=attention_mask.sum()`，CPU 裁掉 padding。
- 后端方案：六段全部 QNN HTP INT8，逐段 compose/execute/free，只保留 hidden 与预计算 mask/cos/sin，峰值约 1 段权重 + 激活。
- **重要坑位：不要接 `final_norm.onnx`。** 官方 pipeline 取 `output_hidden_states=True` 的 `hidden_states[-2]`，即 layer 35 输出、final RMSNorm 之前的值；`layers_30_35` 段输出正好匹配，`final_norm` 会改变数值分布。
- 现状（2026-08-16 evening，设备离线）：
  - **v3 float-IO INT8 已转换并全部构建**：
    `build/qnn_text_encoder_segments_int8_floatio/`（六段 model.bin 各 605,798,400 B）
    → `build/android/jniLibs-text-int8-floatio/arm64-v8a/libqnn_layers_*.so`（六段合计约 3.40 GB）。
  - v3 使用 `--preserve_io datatype`：图输入/输出为 FLOAT_32（QNN 布局
    `hidden[1,2560,512]`、`mask[1,512,512,1]`、`cos/sin[1,128,512]`），
    段边界不再需要 CPU 量化/反量化；内部仍是 W8A8。
  - 校准数据复制到无空格路径 `E:\zimage_calib_qwen3_v2`（QAIRT input_list
    会按空格拆路径，项目路径带空格会解析失败）。
  - 旧 quantized-IO INT8 六段 `.so` 也在 `build/android/jniLibs-text-int8/`。
  - **embedding 段也已完成**：QNN FP16 `build/qnn_text_encoder_embedding_fp16/`，
    Android `libqnn_embedding.so`（778 MB）；I/O `input_ids[1,512]` INT64 →
    `hidden[1,512,2560]` FLOAT32。
  - 预计算资产：`build/text_encoder_precomp/{cos_qnn_f32,sin_qnn_f32,causal_mask_qnn_f32}.raw`
    （cos/sin 为 QNN 布局 `[1,128,512]`，mask 为 `[1,512,512,1]`）。
  - JNI probe `nativeTextEncoderProbe` 已实现并 APK 编译通过：
    embedding → 6 段，跳过 final_norm。


### 1.3 Z-Image Transformer（S3-DiT）— QNN HTP（FP16 / INT4 权重）

- 结构：30 个主 DiT block + 2 noise refiner + 2 context refiner；hidden=3840，30 heads，head_dim=128，FFN hidden=10240，patch=2，f_patch=1，RoPE 三轴 dims=[32,48,48]。
- 参数量：约 **6.15B**；FP32 约 24.6 GB，FP16 约 12.3 GB，INT4 理论约 3.1 GB。
- 序列长度（512x512、无 CFG）：image tokens = 32x32 = 1024；cap 有效长度 L 向上 pad 到 32 的倍数；固定 512 cap 导出时为 1536。
- I/O：
  - latent `[1,16,64,64]`
  - timestep `[1]` = `(1000 - scheduler_t) / 1000`
  - cap_feats `[L,2560]`（可变长）
  - 输出 `noise_pred[1,16,64,64]`；pipeline 送入 scheduler 前会 `noise_pred = -noise_pred`，C++ 必须补取反。
- 后端方案：
  - 首选 QNN HTP：模型加载一次，8 步循环复用 context，每次只换 latent/timestep/cap buffer。
  - 精度：先 FP16 跑通，再用 QAIRT INT4/混合量化（`--pack_4_bit_weights` 或 HTP weight-only）把权重压到约 4 GB。
  - 如 converter 内存峰值过大，30 层可切成 5~6 个 segment 分别转换。
- **必须先解决两个导出问题**：
  1. 现有 `transformer_fp16.onnx` 把 `cap_feats` 固定为 `[512,2560]` 且无 padding mask；官方 pipeline 实际传入裁掉 padding 的可变长度 L。固定 512 会改变 RoPE 位置与 attention 序列。需要重新导出带 `cap_mask[512]` 的版本，或在 QNN 支持动态 L 时导出动态 cap 长度。
  2. FP16 导出在 CPU ONNX Runtime 验证时出现过 NaN，需改用 FP32 参考 / 设备 HTP 做数值验证。
- 量化实验结果（2026-08-16 late night + evening）：
  - 整图 W8A8 失败：layer 18 attention MatMul，`Invalid QnnModel constructed`。
  - 整图 W4A16 失败：layer 18 FFN MatMul，`Tensor size is greater than available memory`。
  - 分段后成功：`frontend + 5x6 层 + final`；W4A8 总 model.bin 约 3.08 GB
    （frontend 347 MB，每组 519 MB，final 1.2 MB）。
  - W8A8 单段约 1.1 GB，W4A8 是移动端推荐配置。
  - **关键修复**：v2 每段校准用独立随机输入，段边界 `unified_in` u8
    编码（约 ±0.1）与真实上游输出（实测 -105 ~ +6236）完全不匹配。
    v3 改用链式校准（`tools/generate_transformer_chain_calibration.py`，
    数据在 `E:\zimage_calib_transformer_chain`）+ `--preserve_io datatype`，
    边界为 FLOAT_16/FLOAT_32，彻底消除边界量化的 encode/decode 与布局截断问题。
  - **v3 已全部转换并构建**：
    `build/qnn_transformer_segments_v3_floatio_w4a8/` →
    `build/android/jniLibs-transformer-v3/arm64-v8a/` 共 7 个 `.so`（约 3.09 GB）。
  - **注意**：QNN converter 把 frontend 常量 `unified_freqs` 输出折叠掉了；
    runtime 必须使用预计算资产
    `build/transformer_freqs_qnn/unified_freqs_f32_nhwc.raw`
    （QNN 布局 `[1,128,1536]` float32，已验证与输入无关）。
  - JNI v3 probe 已实现并编译通过：`nativeTransformerProbe` 串行执行
    frontend → 5 组 layers → final，返回各段耗时与统计；`noise_pred` 的取反
    留给 scheduler 阶段。
- 工具：`tools/export_transformer_segments.py`、`tools/generate_transformer_chain_calibration.py`、`tools/convert_transformer_segments_v3.ps1`。

### 1.4 VAE Decoder — QNN HTP FP16

- 结构：Flux 风格 AutoencoderKL；block_out_channels=[128,256,512,512]，latent_channels=16，4 组 UpDecoderBlock + 注意力 mid block。
- 参数量：83,819,683（约 84M）；ONNX 约 190 MB，现 QNN model lib 199,008,400 B。
- I/O：latent `[1,16,64,64]` → image `[1,3,512,512]`。
- 后端：QNN HTP FP16。当前 `libqnn_vae_gpu.so` 名字是 GPU，但 QNN model backend-agnostic，已在 app 内以 HTP context 成功执行。
- **已修复（2026-08-16 evening）**：`tools/export_vae_onnx.py` 已补 `+ shift_factor`；
  重导出 `build/vae_decoder_shiftfix/vae_decoder.onnx`，ONNX vs PyTorch max diff 7.3e-5。
  QNN FP16 转换 `build/qnn_vae_shiftfix_fp16/`（vae.bin 99,174,400 B），
  Android `.so` `build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so`（99,965,504 B）。
  `zimage_runtime.cpp` 已改为优先加载 `libqnn_vae_shiftfix.so`，旧 GPU 名字库作 fallback。


### 1.5 Scheduler — CPU

- 模型：`FlowMatchEulerDiscreteScheduler`，1000 train timesteps，shift=3.0，无 dynamic shifting。
- Turbo 设置：`guidance_scale=0.0`（无 CFG），8 DiT forwards，512x512。
- 公式（C++ float32）：
  1. `s = linspace(1, 1/N, N)`
  2. `sigma = 3*s / (1 + 2*s)`
  3. append `0` 得到 N+1 个 sigma
  4. 每步 `timestep = sigma*1000`；Transformer 输入 `(1000-t)/1000`
  5. `prev_sample = sample + (sigma_next - sigma) * noise_pred`
  6. 循环结束后 `latents = latents / 0.3611 + 0.1159` 再送 VAE
- 初始 latent：CPU 高斯随机 `[1,16,64,64]`。
- 后端 CPU，每步仅 65,536 float，可忽略。

---

## 2. 后端方案矩阵

| 组件 | 参数量 | 目标权重 | 推荐后端 | 替代 | 风险 |
|---|---|---|---|---|---|
| Tokenizer | 151936 vocab | <10 MB | CPU | - | 低 |
| Qwen3 text encoder | 4.02B | INT8 约 3.4 GB | QNN HTP | CPU fallback / LiteRT | 中 |
| Transformer DiT | 6.15B | FP16 12.3 GB → INT4 约 3.1 GB | QNN HTP | LiteRT GPU / 分段 | 高 |
| VAE decoder | 84M | FP16 约 170 MB | QNN HTP | LiteRT GPU | 低 |
| Scheduler | - | - | CPU | - | 低 |

---

## 3. 设备资源预算（PJZ110）

- 设备 RAM：约 23.6 GB；当前可用约 8.2 GB。
- runtime 目录约 14 GB（主要被 FP32 六段文本编码器占 14.5 GB，需换 INT8）。
- 推荐串行加载：
  - 文本编码：逐段加载约 578 MB 权重 + <20 MB 激活，六段串行。
  - Transformer：常驻 3~6 GB model lib（INT4/FP16 混合），8 步复用。
  - VAE：约 200 MB，输出 3 MB。
  - 预计峰值 6~8 GB，在 24 GB RAM 设备上可行；若用纯 FP16 Transformer（12.3 GB）则需分段或 INT4。

---

## 4. 分阶段实施计划

### Phase A — VAE 修正并产品化 ✅（待设备验证）
1. 修 `tools/export_vae_onnx.py`，补 `+ shift_factor`（已完成）。
2. 重导出并验证 ONNX Runtime vs PyTorch（已完成，max 7.3e-5）。
3. QAIRT FP16 转换 + 构建 `libqnn_vae_shiftfix.so`（已完成）。
4. 运行时已优先加载新 VAE 库；设备验证待办。

### Phase B — 文本编码器 INT8 HTP（构建完成，runtime 待接）
1. 构建六段 INT8 `.so`（已完成，v3 float-IO 与旧 quantized-IO 两套都有）。
2. 设备部署时替换 FP32 六段（14.5 GB → 3.5 GB）。
3. 实现 CPU tokenizer、mask/cos/sin 预计算、六段顺序 graphExecute（待做）。
4. 与 Python `hidden_states[-2]` 对齐，跳过 `final_norm`。

### Phase C — Transformer HTP（v3 构建与 probe 完成，设备数值验证待做）
1. 重新导出带 `cap_mask` 的 ONNX（已完成）。
2. 分段导出 frontend + 5x6 层 + final（已完成）。
3. v3 float-IO W4A8 链式校准转换（已完成，约 3.08 GB）。
4. 构建七段 Android `.so`（已完成，`build/android/jniLibs-transformer-v3/`）。
5. JNI probe `nativeTransformerProbe` 已实现并 APK 编译通过；下一步设备上
   与 ONNX chain 对比，再进入 8 步 scheduler 端到端。

### Phase D — 完整推理链
1. CPU tokenizer → HTP 文本编码 → HTP Transformer x8 → CPU scheduler → HTP VAE → 显示。
2. 各阶段计时与内存统计。
3. 目标：首帧可用 <60s，良好 <15s。

### Phase E（可选）— LiteRT GPU Plan B
1. AI Edge Torch / ONNX→TFLite 转 VAE、Transformer。
2. `CompatibilityList` 检查算子覆盖，跑 `GpuDelegate` 分区。
3. 仅在 QNN HTP Transformer 转换失败或性能不足时启用。

---

## 5. 当前阻塞点清单（2026-08-16 night，设备在连）

1. ~~VAE shift_factor~~ 已修复，HTP 执行成功。
2. ~~文本编码器 INT8 .so~~ v3 float-IO 已构建并在 HTP 分段执行成功（整链待一次干净跑完）。
3. ~~Transformer float-IO 构建~~ 已完成，但：
   - packed W4A8：HTP v79 拒绝 `UFIXED_POINT_4`（datatype 1028 unsupported）。
   - no-pack W4A8（v5）：frontend + layers_00_05 + layers_06_11 已在 HTP 执行；
     随后 LOW_MEMORY 被杀（RSS ~11GB）。
4. **当前正在修**：HTP `QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING` 图配置，
   让 4-bit 范围的 u8 权重在 context binary 里打包；当前 C++ wiring 有 bug，
   所有 `QnnModel_composeGraphs` 返回 8（INVALID_ARGUMENT）。
5. QNN graph handle 不能单独释放，只有 `contextFree` 回收图；若打包仍不够，
   需考虑每段独立 context 或 context binary 缓存。
6. QNN converter 折叠 `unified_freqs`：已用固定资产绕过（需部署到 `<runtime>/assets/`）。
7. CPU tokenizer 与完整 scheduler 链路仍未实现。
