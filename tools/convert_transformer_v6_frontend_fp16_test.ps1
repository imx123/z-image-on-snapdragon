param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe"
)
# Experiment: convert transformer FRONTEND to full FP16 (float_bitwidth 16),
# no quantization, no calibration. Same recipe as the verified VAE FP16 path.
$ErrorActionPreference = "Stop"
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = "E:\projects\zimage on phone\zimage-runtime\qairt-tmp"
$input = "E:\projects\zimage on phone\zimage on phone\build\transformer_segments_v3_floatmask\frontend\model.onnx"
$outDir = "E:\projects\zimage on phone\zimage on phone\build\qnn_transformer_v6_fp16_test\frontend"
New-Item -ItemType Directory -Force $outDir | Out-Null
$modelPrefix = Join-Path $outDir "model"
$log = Join-Path $outDir "conversion.log"
$args = @(
    "--input_network", $input,
    "--output_path", "$modelPrefix.cpp",
    "--input_dim", "latent", "1,16,64,64",
    "--input_dim", "timestep", "1",
    "--input_dim", "cap_feats", "512,2560",
    "--input_dim", "cap_mask", "512",
    "--no_simplification",
    "--float_bitwidth", "16",
    "--float_bias_bitwidth", "16"
)
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& $Python $converter @args *> $log
$code = $LASTEXITCODE
$ErrorActionPreference = $prev
Get-Content $log -Tail 8
if ($code -ne 0) { throw "conversion failed: $code" }
Write-Host "OK: $modelPrefix.cpp / $modelPrefix.bin ($((Get-Item "$modelPrefix.bin").Length) bytes)"