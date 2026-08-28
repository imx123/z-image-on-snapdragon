param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$Onnx = (Join-Path $PSScriptRoot "..\build\vae_decoder_shiftfix\vae_decoder.onnx"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_vae_shiftfix_fp16"),
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Debug
)

# Converts the shift_factor-corrected VAE decoder ONNX to a backend-agnostic
# QNN FP16 model (model.cpp + model.bin). The model is executed on HTP in the
# app; backend selection happens at runtime, not at conversion time.
#
# Android library build (after this script):
#   mkdir $OutputDir\vae_shiftfix; copy vae.cpp/bin as model.cpp/model.bin
#   tools/build_qnn_android_libs.ps1 -QnnSdkRoot E:\qnn -InputRoot $OutputDir -Segments vae_shiftfix

$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
if (-not (Test-Path $Onnx)) { throw "ONNX input not found: $Onnx" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
if (-not $env:QAIRT_TMP_DIR) { throw "QAIRT_TMP_DIR must be set, e.g. E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }

New-Item -ItemType Directory -Force $OutputDir | Out-Null
$inputPath = (Resolve-Path $Onnx).Path
$modelPrefix = Join-Path (Resolve-Path $OutputDir).Path "vae"
$logPath = Join-Path $OutputDir "conversion.log"

Write-Host "Convert VAE ONNX -> QNN FP16 ($modelPrefix.cpp/.bin)"
$args = @(
    "--input_network", $inputPath,
    "--output_path", "$modelPrefix.cpp",
    "--input_dim", "latent", "1,16,64,64",
    "--no_simplification",
    "--float_bitwidth", "16",
    "--float_bias_bitwidth", "16"
)
if ($Debug) { $args += "--debug" }

$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python $converter @args *> $logPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previous
Get-Content $logPath -Tail 15
if ($exitCode -ne 0) { throw "QAIRT conversion failed, exit code $exitCode" }
if (-not (Test-Path "$modelPrefix.cpp") -or -not (Test-Path "$modelPrefix.bin")) { throw "model.cpp/.bin not generated" }
Write-Host "OK: $modelPrefix.cpp / $modelPrefix.bin ($((Get-Item "$modelPrefix.bin").Length) bytes)"
