param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\build\text_encoder_segments_v2"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_text_encoder_segments_int8_floatio"),
    [string]$CalibrationRoot = "E:\zimage_calib_qwen3_v2",
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Force
)

# Qwen3 INT8 with FLOAT graph I/O (--preserve_io datatype). Calibration files
# live under E:\zimage_calib_qwen3_v2 because QAIRT input_list parsing splits
# on spaces and cannot handle the workspace path.
$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }
New-Item -ItemType Directory -Force $OutputDir | Out-Null

foreach ($g in @("layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29","layers_30_35")) {
    $input = Join-Path $SegmentsDir "$g\model.onnx"
    if (-not (Test-Path $input)) { throw "Missing ONNX graph: $input" }
    $out = Join-Path $OutputDir $g
    if ((Test-Path (Join-Path $out "model.bin")) -and -not $Force) { Write-Host "SKIP $g"; continue }
    $list = Join-Path $CalibrationRoot "$g\input_list.txt"
    if (-not (Test-Path $list)) { throw "Calibration missing: $list" }
    New-Item -ItemType Directory -Force $out | Out-Null
    $log = Join-Path $out "conversion.log"
    $args = @(
        "--input_network", (Resolve-Path $input).Path,
        "--output_path", (Join-Path (Resolve-Path $out).Path "model.cpp"),
        "--input_dim", "hidden_states", "1,512,2560",
        "--input_dim", "attention_mask", "1,1,512,512",
        "--input_dim", "cos", "1,512,128",
        "--input_dim", "sin", "1,512,128",
        "--input_list", $list,
        "--weights_bitwidth", "8", "--act_bitwidth", "8", "--bias_bitwidth", "32",
        "--param_quantizer", "tf", "--act_quantizer", "tf",
        "--no_simplification", "--preserve_io", "datatype"
    )
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Python $converter @args *> $log
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    Get-Content $log -Tail 6
    if ($code -ne 0) { throw "Conversion failed: $g (exit $code)" }
    Write-Host "DONE $g -> $out"
}
Write-Host "DONE Qwen3 INT8 float-IO under $OutputDir"
