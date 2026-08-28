param(
    [ValidateSet("W8A8","W4A8")]
    [string]$Mode = "W4A8",
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\build\transformer_segments_v2"),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\build\qnn_transformer_segments"),
    [string]$LayerCalibration = "E:\zimage_calib_transformer_layers\input_list.txt",
    [string]$FrontendCalibration = "E:\zimage_calib_transformer_frontend\input_list_4.txt",
    [string]$FinalCalibration = "E:\zimage_calib_transformer_final\input_list.txt",
    [string]$Python = $env:QAIRT_PYTHON,
    [switch]$Force
)

# Convert each Transformer segment separately. Full-graph QAIRT quantization
# fails at layer 18 ("Tensor size is greater than available memory"), so the
# 30 main layers are exported/converted as five 6-layer groups.
$ErrorActionPreference = "Stop"
if (-not $QnnSdkRoot) { $QnnSdkRoot = "E:\projects\zimage on phone\zimage-runtime\qairt\qairt\2.49.0.260730" }
if (-not $Python) { $Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { "E:\projects\zimage on phone\zimage-runtime\qairt-tmp" }

function Convert-One($InputOnnx, $OutputDir, $InputList, $Dims, [switch]$Pack4) {
    if ((Test-Path (Join-Path $OutputDir "model.bin")) -and -not $Force) { Write-Host "SKIP $OutputDir"; return }
    New-Item -ItemType Directory -Force $OutputDir | Out-Null
    $log = Join-Path $OutputDir "conversion.log"
    $args = @("--input_network", $InputOnnx, "--output_path", (Join-Path $OutputDir "model.cpp"),
              "--input_list", $InputList, "--bias_bitwidth", "32", "--no_simplification")
    foreach ($pair in $Dims) { $args += @("--input_dim", $pair[0], $pair[1]) }
    if ($Mode -eq "W8A8") { $args += @("--weights_bitwidth","8","--act_bitwidth","8") }
    else { $args += @("--weights_bitwidth","4","--act_bitwidth","8","--pack_4_bit_weights") }
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Python $converter @args *> $log
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    Get-Content $log -Tail 8
    if ($code -ne 0) { throw "Conversion failed: $InputOnnx" }
}

$out = Join-Path $OutputRoot $Mode.ToLower()
Convert-One (Join-Path $SegmentsDir "frontend\model.onnx") (Join-Path $out "frontend") $FrontendCalibration @(
    @("latent","1,16,64,64"), @("timestep","1"), @("cap_feats","512,2560"), @("cap_mask","512"))
foreach ($g in @("layers_00_05","layers_06_11","layers_12_17","layers_18_23","layers_24_29")) {
    Convert-One (Join-Path $SegmentsDir "$g\model.onnx") (Join-Path $out $g) $LayerCalibration @(
        @("unified_in","1,1536,3840"), @("unified_freqs","1,1536,128"),
        @("unified_mask","1,1536"), @("adaln_input","1,256"))
}
Convert-One (Join-Path $SegmentsDir "final\model.onnx") (Join-Path $out "final") $FinalCalibration @(
    @("unified_in","1,1536,3840"), @("adaln_input","1,256"))
Write-Host "DONE $Mode under $out"
