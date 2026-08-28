param(
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$Python = "E:\projects\zimage on phone\zimage-runtime\python312\python.exe"
)
# Convert ALL text-encoder segments (embedding + 6 layer groups) to full FP16.
# Same recipe as the verified VAE FP16 path: --float_bitwidth 16, no quantization.
$ErrorActionPreference = "Stop"
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"
$env:QAIRT_TMP_DIR = "E:\projects\zimage on phone\zimage-runtime\qairt-tmp"
$buildRoot = "E:\projects\zimage on phone\zimage on phone\build"
$outRoot = Join-Path $buildRoot "qnn_text_encoder_fp16"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"

function Run-Converter([string]$Name, [string]$InputOnnx, [string]$out, [string[]]$Dims) {
    New-Item -ItemType Directory -Force $out | Out-Null
    $log = Join-Path $out "conversion.log"
    $cmd = @($Python, $converter,
        "--input_network", $InputOnnx,
        "--output_path", (Join-Path $out "model.cpp"),
        "--no_simplification", "--float_bitwidth", "16", "--float_bias_bitwidth", "16",
        "--preserve_io", "datatype")
    foreach ($d in $Dims) { $cmd += @("--input_dim", $d.Split("|")[0], $d.Split("|")[1]) }
    & $cmd[0] @($cmd[1..($cmd.Count-1)]) *> $log
    $code = $LASTEXITCODE
    Get-Content $log -Tail 4
    if ($code -ne 0) { throw "conversion failed: $Name ($code)" }
    if (-not (Test-Path (Join-Path $out "model.cpp"))) { throw "model.cpp missing for $Name" }
    Write-Host "DONE $Name ($((Get-Item (Join-Path $out 'model.bin')).Length) bytes)"
}

Run-Converter "embedding" (Join-Path $buildRoot "text_encoder_segments_v2\embedding\model.onnx") (Join-Path $outRoot "embedding") @("input_ids|1,512")
Run-Converter "layers_00_05" (Join-Path $buildRoot "text_encoder_segments_v2\layers_00_05\model.onnx") (Join-Path $outRoot "layers_00_05") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
Run-Converter "layers_06_11" (Join-Path $buildRoot "text_encoder_segments_v2\layers_06_11\model.onnx") (Join-Path $outRoot "layers_06_11") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
Run-Converter "layers_12_17" (Join-Path $buildRoot "text_encoder_segments_v2\layers_12_17\model.onnx") (Join-Path $outRoot "layers_12_17") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
Run-Converter "layers_18_23" (Join-Path $buildRoot "text_encoder_segments_v2\layers_18_23\model.onnx") (Join-Path $outRoot "layers_18_23") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
Run-Converter "layers_24_29" (Join-Path $buildRoot "text_encoder_segments_v2\layers_24_29\model.onnx") (Join-Path $outRoot "layers_24_29") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
Run-Converter "layers_30_35" (Join-Path $buildRoot "text_encoder_segments_v2\layers_30_35\model.onnx") (Join-Path $outRoot "layers_30_35") @("hidden_states|1,512,2560", "attention_mask|1,1,512,512", "cos|1,512,128", "sin|1,512,128")
$ErrorActionPreference = $prev
Write-Host "ALL TEXT FP16 DONE -> $outRoot"