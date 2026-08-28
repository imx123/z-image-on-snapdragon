param(
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [string]$Onnx = (Join-Path $PSScriptRoot "..\build\transformer_fp16.onnx"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_transformer_gpu"),
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
if (-not $QnnSdkRoot) { $QnnSdkRoot = "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730" }
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
if (-not (Test-Path $Onnx)) { throw "ONNX input not found: $Onnx" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"

New-Item -ItemType Directory -Force $OutputDir | Out-Null
$inputPath = (Resolve-Path $Onnx).Path
$outputPath = Join-Path (Resolve-Path $OutputDir).Path "transformer.cpp"
$logPath = Join-Path $OutputDir "conversion.log"

Write-Host "CONVERT $inputPath -> $outputPath"
Write-Host "QAIRT converter: $converter"
Write-Host "Python: $Python"

# Fixed shapes: latent [1,16,64,64], timestep [1], cap_feats [512,2560]
# backend=GPU is selected by the converter's default op-package settings for
# Adreno; no_simplification matches the proven VAE/segment conversion path.
$args = @(
    "--input_network", $inputPath,
    "--output_path", $outputPath,
    "--input_dim", "latent", "1,16,64,64",
    "--input_dim", "timestep", "1",
    "--input_dim", "cap_feats", "512,2560",
    "--no_simplification"
)
if ($Debug) { $args += "--debug" }

$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python $converter @args *> $logPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previous

Get-Content $logPath -Tail 25
if ($exitCode -ne 0) { throw "QAIRT conversion failed, exit code $exitCode" }
$cpp = Join-Path $OutputDir "transformer.cpp"
$bin = Join-Path $OutputDir "transformer.bin"
if (-not (Test-Path $cpp)) { throw "Conversion completed without CPP output" }
if (-not (Test-Path $bin)) { throw "Conversion completed without BIN output" }
Write-Host "DONE: cpp=$((Get-Item $cpp).Length) bytes, bin=$((Get-Item $bin).Length) bytes"
