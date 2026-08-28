param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe"
)
# FP32 conversion of the 3-layers-per-segment export (v10). 3-layer groups keep
# each FP32 model.bin under the 4GB aarch64 linker relocation limit; 6-layer
# FP32 segments (4.34GB) fail at link with ADR_PREL_PG_HI21 out of range.
$ErrorActionPreference = "Stop"
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = "E:\projects\zimage on phone\zimage-runtime\qairt-tmp"
$buildRoot = "E:\projects\zimage on phone\zimage on phone\build"
$outRoot = Join-Path $buildRoot "qnn_transformer_v10_fp32"

function Convert-One($Name, $InputOnnx, $out, $Dims) {
    New-Item -ItemType Directory -Force $out | Out-Null
    $log = Join-Path $out "conversion.log"
    $args = @("--input_network", $InputOnnx, "--output_path", (Join-Path $out "model.cpp"),
              "--no_simplification", "--float_bitwidth", "32", "--float_bias_bitwidth", "32")
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

$srcRoot = Join-Path $buildRoot "transformer_segments_v10_maskfix_3layer"
Convert-One "frontend" (Join-Path $srcRoot "frontend\model.onnx") (Join-Path $outRoot "frontend") @(
    @("latent","1,16,64,64"), @("timestep","1"), @("cap_feats","512,2560"), @("cap_mask","512"))
foreach ($g in @("layers_00_02","layers_03_05","layers_06_08","layers_09_11","layers_12_14","layers_15_17","layers_18_20","layers_21_23","layers_24_26","layers_27_29")) {
    Convert-One $g (Join-Path $srcRoot "$g\model.onnx") (Join-Path $outRoot $g) @(
        @("unified_in","1,1536,3840"), @("unified_freqs","1,1536,128"),
        @("unified_mask","1,1536"), @("adaln_input","1,256"))
}
Convert-One "final" (Join-Path $srcRoot "final\model.onnx") (Join-Path $outRoot "final") @(
    @("unified_in","1,1536,3840"), @("adaln_input","1,256"))
Write-Host "ALL TRANSFORMER v10 FP32 DONE -> $outRoot"
