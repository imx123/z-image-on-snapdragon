param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\build\transformer_segments_v2"),
    [string]$FrontendOnnx = (Join-Path $PSScriptRoot "..\build\transformer_segments_v3_floatmask\frontend\model.onnx"),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\build\qnn_transformer_segments_v5_w4a8_nopack"),
    [string]$FrontendCalibration = "E:\zimage_calib_transformer_frontend_floatmask\input_list_4.txt",
    [string]$ChainCalibration = "E:\zimage_calib_transformer_chain",
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Force
)

# Transformer v5: float-IO W4A8 without --pack_4_bit_weights.
# HTP v79 rejects QNN_DATATYPE_UFIXED_POINT_4 graph tensors (7004/0x1b5c),
# so 4-bit weights are kept in UFIXED_POINT_8 containers (no pack).
# Frontend ONNX uses a float cap_mask so the final mask path has no bool
# StridedSlice (HTP rejects BOOL_8 StridedSlice with 0xc26/3110).
$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }

function Convert-One($Name, $InputOnnx, $OutputDir, $InputList, $Dims) {
    if ((Test-Path (Join-Path $OutputDir "model.bin")) -and -not $Force) { Write-Host "SKIP $Name"; return }
    New-Item -ItemType Directory -Force $OutputDir | Out-Null
    $log = Join-Path $OutputDir "conversion.log"
    $args = @("--input_network", $InputOnnx, "--output_path", (Join-Path $OutputDir "model.cpp"),
              "--input_list", $InputList, "--bias_bitwidth", "32", "--no_simplification")
    foreach ($pair in $Dims) { $args += @("--input_dim", $pair[0], $pair[1]) }
    $args += @("--weights_bitwidth","4","--act_bitwidth","8","--preserve_io","datatype")
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Python $converter @args *> $log
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    Get-Content $log -Tail 6
    if ($code -ne 0) { throw "Conversion failed: $Name (exit $code)" }
    if (-not (Test-Path (Join-Path $OutputDir "model.bin"))) { throw "model.bin missing for $Name" }
    Write-Host "DONE $Name -> $OutputDir"
}

Convert-One "frontend" $FrontendOnnx (Join-Path $OutputRoot "frontend") $FrontendCalibration @(
    @("latent","1,16,64,64"), @("timestep","1"), @("cap_feats","512,2560"), @("cap_mask","512"))
foreach ($g in @("layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29")) {
    $cal = Join-Path $ChainCalibration "$g\input_list.txt"
    if (-not (Test-Path $cal)) { throw "chain calibration missing: $cal" }
    Convert-One $g (Join-Path $SegmentsDir "$g\model.onnx") (Join-Path $OutputRoot $g) $cal @(
        @("unified_in","1,1536,3840"), @("unified_freqs","1,1536,128"),
        @("unified_mask","1,1536"), @("adaln_input","1,256"))
}
$finalCal = Join-Path $ChainCalibration "final\input_list.txt"
Convert-One "final" (Join-Path $SegmentsDir "final\model.onnx") (Join-Path $OutputRoot "final") $finalCal @(
    @("unified_in","1,1536,3840"), @("adaln_input","1,256"))
Write-Host "DONE Transformer v5 under $OutputRoot"
