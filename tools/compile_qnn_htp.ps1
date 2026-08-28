param(
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [string]$Onnx = (Join-Path $PSScriptRoot "..\build\text_encoder_qnn.onnx"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_text_encoder"),
    [switch]$NoSimplification
)
if (-not $QnnSdkRoot -or -not (Test-Path $QnnSdkRoot)) { throw "QNN_SDK_ROOT is not set or does not exist." }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter was not found: $converter" }
if (-not (Test-Path $Onnx)) { throw "ONNX input was not found: $Onnx" }
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$converterScript = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
$outputPath = Join-Path $OutputDir "text_encoder.cpp"
$converterArgs = @("--input_network", $Onnx, "--output_path", $outputPath, "--input_dim", "input_ids", "1,512", "--input_dim", "attention_mask", "1,512", "--debug")
if ($NoSimplification) { $converterArgs += "--no_simplification" }
& $env:QAIRT_PYTHON $converterScript @converterArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path $outputPath)) { throw "QNN converter exited successfully but did not create output: $outputPath" }
Write-Host "QNN converter output: $outputPath"
