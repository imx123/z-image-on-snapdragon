param(
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\build\text_encoder_segments_v2"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_text_encoder_segments_int8"),
    [string]$CalibrationRoot = "E:\zimage_calib_v2",
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [int]$WeightsBitwidth = 8,
    [int]$ActBitwidth = 8,
    [int]$BiasBitwidth = 32,
    [switch]$Force,
    [switch]$Debug
)

# Int8 conversion for the six Qwen3 layer segments with real calibration data.
# Without an input_list QAIRT skips quantization entirely and emits float32
# weights (~2.26 GiB/segment); with calibration + bitwidth 8 each segment
# should drop to ~670 MB. Usage:
#   . .\tools\setup_qairt_env.ps1
#   .\tools\convert_qwen3_segments_int8.ps1

$ErrorActionPreference = "Stop"
if (-not $QnnSdkRoot) { $QnnSdkRoot = "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730" }
$python = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
if (-not (Test-Path $python)) { throw "QAIRT Python not found: $python" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }

New-Item -ItemType Directory -Force $OutputDir | Out-Null
$groups = Get-ChildItem $SegmentsDir -Directory -Filter "layers_*" | Sort-Object Name
if (-not $groups) { throw "No layer groups found under $SegmentsDir" }

foreach ($group in $groups) {
    $input = Join-Path $group.FullName "model.onnx"
    if (-not (Test-Path $input)) { throw "Missing ONNX graph: $input" }
    $out = Join-Path $OutputDir $group.Name
    $cpp = Join-Path $out "model.cpp"
    $bin = Join-Path $out "model.bin"
    if ((Test-Path $cpp) -and (Test-Path $bin) -and -not $Force) {
        Write-Host "SKIP $($group.Name): existing output"
        continue
    }
    $calibDir = Join-Path $CalibrationRoot $group.Name
    $inputList = Join-Path $calibDir "input_list.txt"
    if (-not (Test-Path $inputList)) { throw "Calibration input_list not found: $inputList" }
    New-Item -ItemType Directory -Force $out | Out-Null
    $inputPath = (Resolve-Path $input).Path
    $outputPath = Join-Path (Resolve-Path $out).Path "model.cpp"
    $listPath = (Resolve-Path $inputList).Path
    Write-Host "CONVERT int8 $($group.Name) with $((Get-Content $listPath).Count) calibration samples"
    $logPath = Join-Path $out "conversion.log"
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $args = @(
        "--input_network", $inputPath,
        "--output_path", $outputPath,
        "--input_dim", "hidden_states", "1,512,2560",
        "--input_dim", "attention_mask", "1,1,512,512",
        "--input_dim", "cos", "1,512,128",
        "--input_dim", "sin", "1,512,128",
        "--input_list", $listPath,
        "--weights_bitwidth", "$WeightsBitwidth",
        "--act_bitwidth", "$ActBitwidth",
        "--bias_bitwidth", "$BiasBitwidth",
        "--param_quantizer", "tf",
        "--act_quantizer", "tf",
        "--no_simplification"
    )
    if ($Debug) { $args += "--debug" }
    & $python $converter @args *> $logPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    Get-Content $logPath -Tail 15
    if ($exitCode -ne 0) { throw "Conversion failed for $($group.Name), exit code $exitCode" }
    if (-not (Test-Path $cpp)) { throw "Conversion completed without CPP output for $($group.Name)" }
    if (-not (Test-Path $bin)) { throw "Conversion completed without BIN output for $($group.Name)" }
    Write-Host "DONE $($group.Name): $((Get-Item $bin).Length) bytes (was 2422548480 fp32)"
}
Write-Host "All Qwen3 layer groups converted (int$WeightsBitwidth) under $OutputDir"
