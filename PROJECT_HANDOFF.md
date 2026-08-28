# Z-Image on Snapdragon 8 Elite — 项目交接文档

> **读者**:没有任何上下文的新 AI/工程师。读完本文档即可继续开发,不需要翻聊天记录。
> 历史流水账:`PROJECT_HANDOFF_ARCHIVE.md`。GPU 后端完整构建文档:`docs/gpu-backend-build.md`。
> **Last Updated**: 2026-08-24
>
> ## 30 秒速览(当前在哪、卡在哪)
> - **目标**:把 Z-Image-Turbo 文生图模型完整跑在 OnePlus 13 手机上(NPU/GPU 推理)。
> - **三大组件已全部在 app 内验证通过**(同进程混合双后端):Transformer=QNN GPU FP32、文本编码器+VAE=QNN HTP FP16。
> - **完整 pipeline 已写完 C++ 代码**(tokenizer→text→8步采样→VAE),构建通过,**但端到端尚未跑出第一张图**。
> - **内存墙已用"app 外预编译"根治**:在设备上用 `qnn-context-binary-generator`(shell 进程,不受 app LMK 限制)把全部 12 个 transformer 段编译成 ctxbin(~25GB)存入外部 segbin 目录;C++ `segBinPath()` 已加外部 fallback。app 内不再有任何编译峰值(compile 是唯一超内存的操作),冷启后点 Generate 应直接 createFromBinary 秒恢复+跑 8 步。
> - **下一步最高优先**:验证外部 segbin 可读(属主权限)→ 删除私有 segbin 旧文件(22:23 app 自存的旧 frontend.ctxbin 可能被优先命中且不可用)→ 点 Generate 跑第一张图。详见 §8/§9。

**Device**: PJZ110 / OnePlus 13 / SM8750(Snapdragon 8 Elite:Adreno 830 GPU + Hexagon HTP v79 NPU)
**Workspace**: `E:\projects\zimage on phone\zimage on phone`(路径含空格!)
**Runtime SDK root**: `E:\projects\zimage on phone\zimage-runtime`
**adb 序列号**: `3B15B100YVR00000`

---

## 1. 项目目标与架构

把 ModelScope **Tongyi-MAI/Z-Image-Turbo**(diffusers `ZImagePipeline`,8 步 Turbo)端侧化:

| 组件 | 参数量 | 后端/精度 | 状态 |
|---|---|---|---|
| Tokenizer(Qwen BPE + chat template) | - | C++ CPU(自研,APK assets 内置词表) | ✅ 已实现,输出与 HF 一致 |
| Qwen3 文本编码器 | 4.0B | QNN **HTP FP16**,7 段(embedding+6×6层组) | ✅ 数值闭环 corr=0.977 |
| Z-Image Transformer(DiT) | 6.15B | QNN **GPU FP32**,12 段(frontend+10×3层+final) | ✅ corr=0.999996(**唯一数值正确**) |
| Scheduler(FlowMatchEuler) | - | C++ CPU(shift=3,8 步) | ✅ 已实现 |
| VAE Decoder | 84M | QNN **HTP FP16**(单图) | ✅ probe 出有限图像 |

- **为什么 Transformer 是 GPU FP32**:全 FP16 在 QNN 的 MatMul/FC attention 投影处累加溢出成 inf/NaN(PC ORT 同模型有限,证明是 QNN 实现问题)。FP32 完全消除(corr=0.999996)。HTP 对 FP32 图仍走内部 FP16 路径也会 NaN,所以 v10 是 **GPU 专用产物**。
- **3-layer 分段原因**:6-layer FP32 段 model.bin 达 4.34GB,超过 aarch64 链接器相对寻址 4GB 上限,ld.lld 报 out of range;3-layer 段 2.17GB 可链接。
- 目标性能:512×512、8 步、首帧 <60s。

---

## 2. 环境与依赖

### 2.1 关键路径

| 项 | 路径 |
|---|---|
| 项目根 | `E:\projects\zimage on phone\zimage on phone` |
| 运行时/SDK/模型根 | `E:\projects\zimage on phone\zimage-runtime` |
| QAIRT SDK | `...\zimage-runtime\qairt\qairt\2.49.0.260730` |
| QAIRT 无空格 junction | `E:\qnn`(**所有构建脚本显式传 `-QnnSdkRoot E:\qnn`**,否则带空格路径被拆碎) |
| QAIRT Python | `E:\projects\zimage on phone\zimage-runtime\python312\python.exe`(torch/diffusers/transformers/onnxruntime 已装) |
| Gradle home | `E:\projects\zimage on phone\zimage-runtime\gradle-home` |
| Gradle 8.11.1 | `C:\Users\Max\.gradle\wrapper\dists\gradle-8.11.1-bin\bpt9gzteqjrbo1mjrsomdt32c\gradle-8.11.1\bin\gradle.bat` |
| JDK 17(**必须用这个**,别用 Android Studio JBR) | `C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot` |
| APK | `android/app/build/outputs/apk/debug/app-debug.apk` |

### 2.2 Git Bash 环境变量(每个新 shell 先设)

```bash
export QAIRT_PYTHON='E:\projects\zimage on phone\zimage-runtime\python312\python.exe'
export QAIRT_TMP_DIR='E:\projects\zimage on phone\zimage-runtime\qairt-tmp'
export PYTHONPATH='E:/qnn/lib/python'          # 仅 converter 需要
export GRADLE_USER_HOME='E:\projects\zimage on phone\zimage-runtime\gradle-home'
export JAVA_HOME='C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot'
export MSYS_NO_PATHCONV=1                       # 所有含 /data/... 的 adb 命令前必须设
```

### 2.3 构建/部署/看日志(全部已验证可用)

```bash
# 构建 APK(~5s 增量)
cd '/e/projects/zimage on phone/zimage on phone/android'
'/c/Users/Max/.gradle/wrapper/dists/gradle-8.11.1-bin/bpt9gzteqjrbo1mjrsomdt32c/gradle-8.11.1/bin/gradle.bat' assembleDebug

# 安装启动
export MSYS_NO_PATHCONV=1
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.example.zimage
adb shell am start -n com.example.zimage/.MainActivity

# C++ 日志(最重要)/进度
adb shell 'run-as com.example.zimage cat files/zimage-runtime/jni.log'
adb shell 'run-as com.example.zimage cat files/zimage-runtime/progress.txt'

# 切后端(htp|gpu|cpu;gpu=当前生产配置)
adb shell "printf 'gpu' > /storage/emulated/0/Android/data/com.example.zimage/files/probe/backend.txt"
```

### 2.4 部署 runtime 模型文件

```bash
cd '/e/projects/zimage on phone/zimage on phone'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/deploy_android_runtime.ps1
# GPU/FP32 时覆盖:
powershell.exe -File tools/deploy_android_runtime.ps1 `
  -TransformerLibRoot build\android\jniLibs-transformer-v10-fp32 `
  -TextEncoderLibRoot build\android\jniLibs-text-fp16
```
App 首启会把 external 下 .so 按"大小不同才复制"拷进私有目录;**同大小重建需先删私有旧文件**:
```bash
adb shell 'run-as com.example.zimage find files/zimage-runtime/lib -name "*.so" -delete'
```

---

## 3. 当前代码状态(`android/app/src/main/cpp/zimage_runtime.cpp`,~1050 行)

### 3.1 双后端架构(✅ 2026-08-23 打通)

- `Runtime{ QnnSet main; QnnSet hp; }`:main=transformer 后端(backend.txt 决定,gpu/cpu/htp);hp=固定 HTP(VAE+text 专用)。`backendKind`:0=HTP 单后端,1=CPU,2=GPU(生产)。
- **初始化顺序关键**(`nativeCreate`):非 HTP 模式下**先创建 hp(HTP2)+compose VAE,再创建 main(GPU)**。此时 HTP 是进程里唯一后端,VAE 图构建正常。
- **三处致命 bug 已修(2026-08-23,曾导致双 backend 全部 6001/compose=4)**:`composeVae`/`composeSegment`/`runGraph` 曾使用全局 `r.backend/r.api/r.context`,在双 backend 下打到错误后端上。**规则:所有图操作必须用图所属的 `QnnSet& s` 的句柄(s.backend/s.api/s.context)**。修复后混合模式全通:GPU 12 段 + HTP text 7 段 + VAE 同进程同时工作。
- `GraphInfo` ABI 必须与 SDK `share/QNN/converter/jni/QnnWrapperUtils.hpp` 完全一致:`{graph, graphName, inputTensors, numInputTensors, outputTensors, numOutputTensors}`。字段顺序错 → graphName 指针被当 handle → 100% `6001 INVALID_HANDLE`(2026-08-19 大坑,已修)。

### 3.2 完整 pipeline(✅ 代码完成,⚠️ 未跑通端到端)

`nativeGenerate(prompt)` 流程:
1. **Tokenizer**(`namespace zpipe`):自研 GPT2/Qwen byte-level BPE。数据文件 `assets/qwen_vocab.tsv`(id\ttoken,2.6MB)+`assets/qwen_merges.txt`(空格分隔对,1.8MB),由 PC 从 HF tokenizer.json 导出并 adb push 到设备 external assets。chat template 硬编码:`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`。pretokenize 规则=GPT2 正则近似("空格跟随后面的整个单词";数字 3 个一组;其他逐符号);byte-unicode 映射(空格→Ġ,\n→Ċ,ASCII 直通,高位字节→U+0100+byte)。**PC 端已用相同算法+数据验证与 HF 输出完全一致**(a red paper lantern → `[151644,872,198,64,2518,5567,73165,151645,198,151644,77091,198,...]`)。
   - ⚠️ 注意:BPE 必须按 UTF-8 字符切分初始符号,不能按字节;换行必须先转 Ċ 再查词表。
2. **Text encoder**(`pipeTextEncoder`):ids int64[512] → embedding → 6 组 layers(hidden NHWC [1,2560,512]) → **转置回 [512,2560]** 作为 cap_feats。attention mask 构造 [512,512]:causal + padding(-inf)。
3. **Scheduler**(`buildSigmas`):sigmas=linspace(1,1/N,N) → shift 变换 `s*3/(1+2s)`;模型时间输入 `tNorm=1-sigma`;Euler 更新 `latent += dt * (-noise_pred)`(**noise_pred 取反在这里做了**)。seed 固定 42(std::mt19937_64 + normal_distribution)。
4. **Transformer**(`pipeTransformer`):frontend→10 层组→final,每步全段重编译(unified 转置 [1536,3840]→[3840,1536];freqs 从 assets 读)。
5. **VAE**(`pipeVaeDecode`):latent CHW→QNN NHWC [1,64,64,16],输出 [1,512,512,3] float→RGB uint8。**shiftfix 图内已含 /0.3611+0.1159,C++ 直接喂原始 latent,不要二次归一化**(2026-08-24 刚修掉这个 bug)。

### 3.3 已知未解决问题(pipeline 卡点)

- **P0|transformer 段重复编译**:pipeTransformer 每次调用都 contextCreate+composeSegment 全部 12 段,GPU FP32 段首次 finalize 约 30-60s/段。probe 模式一次性跑 14 段没问题,但 pipeline 8 步×12 段不可接受。**修法:compose 一次缓存 GraphInfo/context,循环只做 graphExecute**。注意内存:12 段同时持有约 23GB 权重 mmap,需要评估(权重是 mmap 的,常驻代价可能可接受;context 元数据小)。
- **P1|generate 过程进程被杀一次**(2026-08-23 夜):日志停在 text_layers_18_23 编译中,pid 消失无 tombstone,当时 RSS ~5GB、lowmemorykiller 报 freeMemory ~1.1GB。怀疑与 text 段编译内存峰值叠加有关;待复现确认。缓解方向:P0 的缓存改造会大幅减少反复 compose/free;必要时 text 各段也做常驻缓存。
- 启动时 MainActivity 会自动跑一次 `generate("feasibility-probe")` 作为 VAE 探针(现在会触发完整 pipeline!)——调试时可先注释掉或改回零 latent VAE-only probe,避免每次启动都跑 8 步。

---

## 4. 模型与推理链契约(必读,错一个就全盘垃圾)

### 4.1 数值语义

- 文本编码器取 `hidden_states[-2]`(第 35 层输出,final RMSNorm 之前)。
- **noise_pred 进 scheduler 前取反**(C++ 在 euler 步里 `-noise[k]`)。
- VAE shiftfix ONNX 内部已做 `latents/0.3611+0.1159`;外部喂原始 latent。
- 所有 FP16/FP32 转换必须加 `--preserve_io datatype`(否则 I/O dtype 错位,输出全垃圾)。

### 4.2 QNN I/O 布局(QNN 是 NHWC!)

| Tensor | ONNX 逻辑 | QNN 图 I/O |
|---|---|---|
| Transformer latent | [1,16,64,64] CHW | [1,64,64,16](perm 0,3,1,2)FP32(v10) |
| unified(frontend 出) | [1,1536,3840] | 同左;layer 入需 transpose→[1,3840,1536] |
| freqs(常量折叠) | [1,1536,128] | assets/unified_freqs_f32_nhwc.raw [1,128,1536] FP32 |
| Qwen hidden | [1,512,2560] | [1,2560,512](NHWC) |
| cos/sin | [1,512,128] | [1,128,512] FP32 |
| Qwen mask | [1,1,512,512] | [1,512,512,1] FP32(causal+pad,-inf) |
| cap_feats/mask | [512,2560]/[512] | 同左,dtype 按图动态(FP32/FP16) |

模型结构:Qwen3 36 层 hidden 2560(32Q/8KV head_dim128);DiT 30 主层+2 noise refiner+2 context refiner hidden 3840 30 heads FFN10240 patch2;VAE Flux 风格。scheduler 配置(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=false)来自模型仓 scheduler/scheduler_config.json;VAE scaling_factor=0.3611 shift_factor=0.1159 来自 vae/config.json。

---

## 5. 有效产物清单(build/ 下)

### 5.1 Android .so(部署到设备 external zimage-runtime/lib/arm64-v8a/)

| 目录 | 内容 | 状态 |
|---|---|---|
| `build/android/jniLibs-transformer-v10-fp32/arm64-v8a/` | v10 FP32 全 12 段(~23GB) | ✅ 生产(transformer) |
| `build/android/jniLibs-text-fp16/arm64-v8a/` | text embedding+6 组(~xGB) | ✅ 生产(text) |
| `build/android/jniLibs/arm64-v8a/libqnn_vae_shiftfix.so` | VAE 修正版 | ✅ 生产(VAE) |
| `build/android/apk-jniLibs/arm64-v8a/` | 仅 libQnnHtpV79Stub.so(打进 APK) | ✅ |
| jniLibs-transformer-v6-fp16 / -v7-gpu-fp16 / -v9-fp32-test | 历史/实验 | 弃用或仅参考 |

QAIRT 运行库(libQnnHtp.so/libQnnGpu.so/QnnModel*.so 等 122 个)也在同一目录,来源 SDK `lib/aarch64-android`。

### 5.2 设备资产(external zimage-runtime/assets/,app 复制到私有)

- `unified_freqs_f32_nhwc.raw` [1,128,1536] FP32
- `cos_qnn_f32.raw`/`sin_qnn_f32.raw` [1,128,512]、`causal_mask_qnn_f32.raw`
- `qwen_vocab.tsv`(2.6MB)、`qwen_merges.txt`(1.8MB)— 自研 tokenizer 数据,PC 源在 `android/app/src/main/assets/`
- probe 输入(PC):`build/probe_inputs_v9_fp32/`(FP32)、`ort_reference/` 有前端 ORT 参考输出

### 5.3 QNN 转换目录

- `build/qnn_transformer_v10_fp32/`(frontend+layers_00_02..27_29+final)✅
- `build/qnn_text_encoder_fp16/`、`build/qnn_vae_shiftfix_fp16/` ✅
- v6/v7/v9 与旧 W4A8 目录:历史/废弃。

---

## 6. 高频坑位清单(按踩坑频率排序)

1. **路径含空格**:构建脚本传 `-QnnSdkRoot E:\qnn`(junction);校准/input_list 路径绝不能有空格(用 `E:\zimage_calib_*`)。
2. **MSYS_NO_PATHCONV=1**:git-bash 里所有含 `/data/...`、`/storage/...` 的 adb 命令不加就报 no such file。
3. **patch 工具会把整文件行尾转 CRLF**:每次改完 cpp 必须 `tr -d '\r' < f > /tmp/x && mv /tmp/x f` 归一化,否则 NDK 编译可能出错。
4. **图操作句柄必须来自图所属 QnnSet**:混用全局 Runtime 句柄 = 双 backend 下 compose=4/execute=6001(§3.1 三处 bug 教训)。
5. **GraphInfo 字段顺序**错了就是 6001(§3.1)。
6. **每段独立 context + 执行后 contextFree**:共享 context 会 OOM(RSS 峰值 5.6GB 教训)。graph 不能单独 free。
7. **FP16 MatMul 溢出**:attention 投影处 inf/NaN;正解 FP32(v10)。不要试图"修 FP16"。
8. **OpenCL 不能打进 APK**:Manifest `uses-native-library libOpenCL.so required=false`,库用系统 vendor 的;打进包会 clGetPlatformIDs=-1001。
9. **qnn-onnx-converter 没有 --backend 参数**,backend 是运行时选的;转换产物 .so 与后端无关。
10. 构建 .so:objcopy 用相对路径 `obj/binary/foo.raw`;必须编 `linux/QnnModelPal.cpp`。
11. 多输入 qnn-net-run input_list 单行 `name:=file` 空格分隔。
12. HTP 不支持 bool Cast/StridedSlice(bool 子图要用 onnx_floatize_where.py 消除);v79 不支持 packed 4-bit。
13. 设备 CLI 快速验证(不经 app):
    ```bash
    adb push <qnn-net-run> <libQnnGpu.so 或 libQnnHtp.so> <model.so> /data/local/tmp/gputest/
    adb shell 'cd /data/local/tmp/gputest && ADSP_LIBRARY_PATH=. LD_LIBRARY_PATH=. ./qnn-net-run --model X.so --backend libQnnHtp.so --input_list il.txt --output_dir out --use_native_input_files'
    ```
    (ADSP_LIBRARY_PATH 必须指向含 libQnnHtpV79Skel.so 的目录,否则 transport 14001。)

---

## 7. 验证基准(改动后回归对照)

| 项 | 基准值 |
|---|---|
| v10 GPU frontend unified | min=-61.46 max=739.95 mean≈0.0412(与 ORT corr=0.99999998) |
| v10 GPU final noise_pred | min=-6.66 max=6.72 mean≈-0.06(vs ORT corr=0.999996,maxabs 0.019) |
| text hidden[-2] | corr=0.977 rmse=2.08 vs ORT;末段 min≈-702 max≈2506 |
| VAE probe(随机 latent) | 输出有限(min~-0.54 max~0.33);GPU FP16 版才是 NaN |
| tokenizer(HF 对照) | a red paper lantern → 前 12 id 见 §3.2;feasibility-probe → feas/ibility/-/probe 四 token |
| 性能参考 | frontend exec ~820ms;text 每组 ~1s;GPU layer 段 exec ~2s;VAE ~?ms |

---

## 8. 下一步(按优先级)

> **2026-08-28 更新:M3 UI 重构完成 + 设置界面 + 持久化历史。** 测试版界面已重写为 Material 3(动态取色/五层布局/状态机/环心计时/诊断sheet带复制导出/全屏预览/保存分享)。设置界面:采样步数(1-16)+随机种子开关(seed=0真随机)+固定种子值+分辨率(模型固定512,占位)。历史:图片存 app 私有目录 `files/zimage-runtime/images/`(时间戳命名 PNG + history.json 索引),重启不丢;长按缩略图出 保存/分享/删除 菜单;恢复上次结果不再回填画布(仅历史条可见)。全部功能真机验证通过。

1. **P0 ✅ 已闭环:端到端出图**。GPU 内存泄漏的根治方案=**每 4 段 release 后销毁重建 GPU backend**(backendFree+deviceFree+dlclose→重建),实测 RSS 10GB→200MB 瞬间回落。第一张 512×512 图 583.7s(transformer 558.9s,含 24 次 rebuild 开销)。
2. **P3(进行中):图片落盘/显示**。已加 `r->root+"/output.ppm"` 写入(P6 格式,RGB 交错与 VAE HWC 输出直接匹配);待拉回 PC 验证图像内容正确性,然后接 UI 显示。
3. **性能优化(新)**:①rebuild 频率 4→5 段(峰值 ~12GB 边缘)或减少 rebuild 开销;②kernel repo 磁盘缓存(QNN_GPU_CONTEXT_CONFIG_OPTION_KERNEL_REPO_DIR)让 backend 重建后 kernel 从磁盘秒载;③首帧 10 分钟 → 目标 3-5 分钟。
4. **P2:占用率面板** — 已探明 ColorOS 锁死全部系统节点(/proc/stat、sysfs、kgsl 连 shell 都拒),唯一可读 /proc/self/* 与 cpufreq。已放弃(用户确认恢复原状)。
5. P4(可选):text 转 FP32 提精度(约 16GB 存储,3-layer 分段 ×12)。

---

## 9. 本会话(08-23/24)完整变更记录

### 08-24 内存攻坚(重要)
- **LMK 根因数据**:双缓存常驻 19 段 context → RSS 11GB → ColorOS 杀进程(`ApplicationExitInfo reason=LOW_MEMORY`)。前台 app 实测 RSS 上限约 8-9GB。
- **解法:QNN context binary**(官方 API,GPU 后端支持):
  - 每段 compile 后 `S.api.contextGetBinarySize/contextGetBinary` 导出二进制到 `files/zimage-runtime/segbin/<段名>.ctxbin`,**立即 contextFree**;
  - 执行时 `createFromBinary` 恢复单段 → `graphRetrieve(ctx,"model")` 取图句柄 → execute → free。任意时刻只有 1 个 transformer/text 段在内存。
  - ⚠️ 二进制恢复后的 graph 句柄必须用 `graphRetrieve` 获取,**不能**再调 composeGraphs(会失败)。
  - ⚠️ text 段磁盘文件名是 `libqnn_layers_*.so`(没有 text_ 前缀),而缓存名是 `text_layers_*`;cacheLoadSegment 已做名字映射。
- 启动自动探针已改回零 latent VAE-only(`nativeVaeProbe`),不再每次启动全 pipeline 触发 OOM。
- MainActivity 的 Generate 按钮改为后台线程调用(原在 UI 线程同步跑会 ANR)。
- 实测:probe 全 14 段跑完 RSS 仅 ~365MB,进程稳定。

### 08-26/27 内存泄漏诊断与「重建 backend」终局方案(端到端出图)

**诊断链**(每步都有实测数据,勿再走弯路):
1. **RSS 日志**(readSelfRssMB 读 /proc/self/status):每段 restore 后 +~2GB,release 后只回落一半,残留累积,4 段撞 12GB → LMK 杀。
2. **smaps_rollup 归因**:泄漏全是 **file-backed 页**(anon 不涨反降),即 GPU 权重走 kgsl **dma-buf(/dev/dma_heap/system)** 分配,contextFree 后 kgsl 驱动把 buffer 放缓存池不归还。
3. **排除项**:①sleep(10) 无效(非异步延迟);②madvise(MADV_DONTNEED) 对 HTP text 段有效(立即回落)、对 GPU 段无效(dma-buf 是设备映射,忽略 DONTNEED);③`QNN_GPU_BACKEND_CONFIG_OPTION_WEIGHT_SHARING_ENABLED` backend 配置 → libQnnGpu.so 内部 SIGSEGV(driver bug,禁用!)。⚠️ QNN custom config 官方构造:外层 `QnnBackend_Config_t*` 数组 **NULL 终止**,内层 customConfig 是**单个 struct 指针**(docs saver_backend 示例),不是 UNDEFINED 哨兵。
4. **终局方案(✅ 验证成功)**:每 free 4 段后 `rebuildGpuBackend()`=r.main.release(backendFree+deviceFree+logFree+dlclose libQnnGpu.so)→makeBackend 重建。**kgsl dma-buf 缓存挂在 backend/lib 生命周期上,释放瞬间 RSS 10GB→200MB**。8 步全程 24 次重建,每次 ~1-2s。
5. **第一张图**:prompt "a red paper lantern" → 红纸灯笼(语义完全正确)。583.7s 总耗时(transformer 558.9s)。图片 PPM 落盘 `r->root+"/output.ppm"`(P6 格式,与 VAE HWC RGB 输出直接匹配),拉回 PC 转 PNG 验证。

**性能优化方向**(下次):①kernel repo 磁盘缓存 `QNN_GPU_CONTEXT_CONFIG_OPTION_KERNEL_REPO_DIR`(重建后 kernel 从磁盘秒载,省反序列化);②rebuild 间隔 4→5 段;③首帧 583s → 目标 3-5 分钟。

**部署教训**:第三方安装器(com.rosan.installer)重装会清空外部数据目录(全部 libs/assets/segbin/backend.txt)。恢复清单:116 个 libs(含 hexagon-v79 skel!缺失→HTP deviceCreate=14001)+8 assets(tokenizer 2 文件+freqs/cos/sin/mask)+backend.txt=gpu+12 个 ctxbin(从 /data/local/tmp 恢复,重装不清)。注意 v79 目录的 libQnnSystem.so 是裁剪版 217KB,必须用 aarch64-android 的完整版 4,072,160 字节,否则 binaryInfo 全失败→"missing input tensor"。

## 9. 本会话(08-23/24)完整变更记录

### 08-24 深夜:预编译 ctxbin 方案(当前主路径)
- **问题升级**:transformer 12 段在 app 内 compile(compose+finalize)时:①每段编译后立即 free 的 step-restore 模式仍把 **frontend 单段 compile 峰值推到系统上限**(2.9GB 权重 + GPU context),ColorOS LMK 在"编译 transformer 段(一次性)"阶段杀进程;②fast path(createFromBinary 恢复)原实现把 context 常驻,12 段全驻留 → RSS 11GB 被杀(已改:verify 后立即 free)。
- **终局方案:app 外预编译**。用 SDK 工具 `qnn-context-binary-generator`(设备端 aarch64 版,作为 shell 进程运行,**不受 app LMK 限制**)对每个 transformer 段 .so 生成 `.ctxbin.bin`;完成后 shell cp 到外部目录 `/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime/segbin/<name>.ctxbin`。
- **C++ 已改** `segBinPath()`:私有 segbin 找不到时 fallback 读外部 segbin(预编译产物)。app 启动后 generate 只需 `createFromBinary` 秒恢复,不再有编译峰值。
- **12 段全部预编译完成并部署**:frontend 2.94GB + layers×10 2.18GB 各 + final 5.1MB,共 ~25GB,位于外部 segbin。设备 LATEST 状态:私有 segbin 里也有 frontend.ctxbin(22:23 app 自存)等旧文件,外部有 23:19 的预编译版。
- 注意:外部目录属 shell 用户(`-rw-rw---- shell shell`),app(u0_a554)能否读待验证——若不能,需 chmod 或改属主。

### 08-24 内存攻坚(重要)
- **LMK 根因数据**:双缓存常驻 19 段 context → RSS 11GB → ColorOS 杀进程(`ApplicationExitInfo reason=LOW_MEMORY`)。前台 app 实测 RSS 上限约 8-9GB。
- **解法:QNN context binary**(官方 API,GPU 后端支持):
  - 每段 compile 后 `S.api.contextGetBinarySize/contextGetBinary` 导出二进制到 `files/zimage-runtime/segbin/<段名>.ctxbin`,**立即 contextFree**;
  - 执行时 `createFromBinary` 恢复单段 → `graphRetrieve(ctx,"model")` 取图句柄 → execute → free。任意时刻只有 1 个 transformer/text 段在内存。
  - ⚠️ 二进制恢复后的 graph 句柄必须用 `graphRetrieve` 获取,**不能**再调 composeGraphs(会失败)。
  - ⚠️ binary 恢复的段 `freeGraphs` 必须置 null(调用会 abort in free,见 tombstone 08-24 19:14);releaseSegment 走 delete 分支。
  - ⚠️ binary 恢复段的张量元数据必须用 `libQnnSystem.so` 的 `QnnSystemInterface_getProviders()` → `provs[0]->v1_13`(union 成员名按 API 版本,是 `v1_13` 不是 `systemContextApiV1_13`)→ `systemContextCreate/systemContextGetBinaryInfo` 填充;**BinaryInfo version=3**(`QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3`),不是 V1!C++ 已兼容 V1/V3。getBinaryInfo 返回后 **不要 free systemContext**(张量指针由它持有)。
  - ⚠️ text 段磁盘文件名是 `libqnn_layers_*.so`(没有 text_ 前缀),而缓存名是 `text_layers_*`;cacheLoadSegment 已做名字映射。
  - ⚠️ `cacheSaveBinaryImpl`/`saveSegmentBinary` 用 `contextGetBinarySize/GetBinary`;generator 工具输出名会自带 `.bin` 后缀(`*.ctxbin.bin.bin` 需改名)。
- 启动自动探针已改回零 latent VAE-only(`nativeVaeProbe`),不再每次启动全 pipeline 触发 OOM。
- MainActivity 的 Generate 按钮改为后台线程调用(原在 UI 线程同步跑会 ANR)。
- 实测:probe 全 14 段跑完 RSS 仅 ~365MB,进程稳定。

## 10. 参考

- GPU 构建详解:`docs/gpu-backend-build.md`(v10 从转换到设备验证全套命令)
- 历史流水账:`PROJECT_HANDOFF_ARCHIVE.md`;组件方案:`docs/backend-plan.md`;旧 I/O 契约:`docs/qnn-v3-io-contract.md`
- diffusers 参考:`...\zimage-runtime\python312\Lib\site-packages\diffusers\pipelines\z_image\pipeline_z_image.py`(get_default_z_image_sigmas/encode_prompt/去噪循环都在这)
