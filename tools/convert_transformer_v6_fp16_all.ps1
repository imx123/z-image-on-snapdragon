param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe"
)
# Batch: convert ALL transformer segments (frontend + 5 layers + final) to full FP16.
# --float_bitwidth 16, no quantization, no calibration (same as VAE FP16 recipe).
$ErrorActionPreference = "Stop"
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = "E:\projects\zimage on phone\zimage-runtime\qairt-tmp"
$buildRoot = "E:\projects\zimage on phone\zimage on phone\build"
$outRoot = Join-Path $buildRoot "qnn_transformer_v6_fp16"

function Convert-One($Name, $InputOnnx, $out, $Dims) {
    New-Item -ItemType Directory -Force $out | Out-Null
    $log = Join-Path $out "conversion.log"
    $args = @("--input_network", $InputOnnx, "--output_path", (Join-Path $out "model.cpp"),
              "--no_simplification", "--float_bitwidth", "16", "--float_bias_bitwidth", "16",
              "--preserve_io", "datatype")
    foreach ($pair in $Dims) { $args += @("--input_dim", $pair[0], $pair[1]) }
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Python $converter @args *> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Get-Content $log -Tail 4
    if ($code -ne 0) { throw "conversion failed: $Name ($code)" }
    if (-not (Test-Path (Join-Path $out "model.cpp"))) { throw "model.cpp missing for $Name" }
    Write-Host "DONE $Name ($((Get-Item (Join-Path $out 'model.bin')).Length) bytes)"
}

Convert-One "frontend" (Join-Path $buildRoot "transformer_segments_v6_maskfix\frontend\model.onnx") (Join-Path $outRoot "frontend") @(
    @("latent","1,16,64,64"), @("timestep","1"), @("cap_feats","512,2560"), @("cap_mask","512"))
foreach ($g in @("layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29")) {
    Convert-One $g (Join-Path $buildRoot "transformer_segments_v6_maskfix\$g\model.onnx") (Join-Path $outRoot $g) @(
        @("unified_in","1,1536,3840"), @("unified_freqs","1,1536,128"),
        @("unified_mask","1,1536"), @("adaln_input","1,256"))
}
Convert-One "final" (Join-Path $buildRoot "transformer_segments_v6_maskfix\final\model.onnx") (Join-Path $outRoot "final") @(
    @("unified_in","1,1536,3840"), @("adaln_input","1,256"))
Write-Host "ALL TRANSFORMER FP16 DONE -> $outRoot"