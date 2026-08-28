param(
    [ValidateSet("W8A8","W4A16","W4A8")]
    [string]$Mode = "W8A8",
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [string]$Onnx = (Join-Path $PSScriptRoot "..\build\transformer_fp16.onnx"),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\build\qnn_transformer_htp"),
    [string]$CalibrationList = "E:\zimage_calib_transformer\input_list_3.txt",
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Debug
)

# Convert the fixed-shape Z-Image Transformer ONNX to a quantized QNN model.
# NOTE: qnn-onnx-converter in QAIRT 2.49 does NOT accept --backend. Backend is
# selected later by qnn-context-binary-generator with libQnnHtp.so.
$ErrorActionPreference = "Stop"
if (-not $QnnSdkRoot) { $QnnSdkRoot = "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730" }
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
if (-not (Test-Path $Onnx)) { throw "ONNX input not found: $Onnx" }
if (-not (Test-Path $CalibrationList)) { throw "Calibration input_list not found: $CalibrationList" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }

$outDir = Join-Path $OutputRoot $Mode.ToLower()
New-Item -ItemType Directory -Force $outDir | Out-Null
$inputPath = (Resolve-Path $Onnx).Path
$outputPath = Join-Path (Resolve-Path $outDir).Path "transformer.cpp"
$logPath = Join-Path $outDir "conversion.log"
$listPath = (Resolve-Path $CalibrationList).Path

Write-Host "CONVERT $Mode $inputPath -> $outputPath"
Write-Host "calibration=$listPath"

$args = @(
    "--input_network", $inputPath,
    "--output_path", $outputPath,
    "--input_dim", "latent", "1,16,64,64",
    "--input_dim", "timestep", "1",
    "--input_dim", "cap_feats", "512,2560",
    "--input_list", $listPath,
    "--bias_bitwidth", "32",
    "--no_simplification"
)
switch ($Mode) {
    "W8A8" {
        $args += @("--weights_bitwidth","8","--act_bitwidth","8")
    }
    "W4A16" {
        $args += @("--weights_bitwidth","4","--act_bitwidth","16",
                   "--pack_4_bit_weights","--keep_weights_quantized",
                   "--restrict_quantization_steps","-0x8000 0x7F7F")
    }
    "W4A8" {
        $args += @("--weights_bitwidth","4","--act_bitwidth","8",
                   "--pack_4_bit_weights")
    }
}
if ($Debug) { $args += "--debug" }

$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python $converter @args *> $logPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previous

Get-Content $logPath -Tail 30
if ($exitCode -ne 0) { throw "QAIRT $Mode conversion failed, exit code $exitCode" }
$cpp = Join-Path $outDir "transformer.cpp"
$bin = Join-Path $outDir "transformer.bin"
if (-not (Test-Path $cpp)) { throw "Conversion completed without CPP output" }
if (-not (Test-Path $bin)) { throw "Conversion completed without BIN output" }
Write-Host "DONE $Mode cpp=$((Get-Item $cpp).Length) bin=$((Get-Item $bin).Length)"
