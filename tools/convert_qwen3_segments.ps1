param(
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\build\text_encoder_segments_v2"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\build\qnn_text_encoder_segments"),
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
if (-not $QnnSdkRoot) { $QnnSdkRoot = "E:\zimage-runtime\qairt\qairt\2.49.0.260730" }
$python = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "E:\zimage-runtime\python312\python.exe" }
$converter = Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) { throw "qnn-onnx-converter not found: $converter" }
if (-not (Test-Path $python)) { throw "QAIRT Python not found: $python" }
$env:PYTHONPATH = Join-Path $QnnSdkRoot "lib\python"

New-Item -ItemType Directory -Force $OutputDir | Out-Null
$groups = Get-ChildItem $SegmentsDir -Directory -Filter "layers_*" | Sort-Object Name
if (-not $groups) { throw "No layer groups found under $SegmentsDir" }

foreach ($group in $groups) {
    $input = Join-Path $group.FullName "model.onnx"
    if (-not (Test-Path $input)) { throw "Missing ONNX graph: $input" }
    $out = Join-Path $OutputDir $group.Name
    $cpp = Join-Path $out "model.cpp"
    $bin = Join-Path $out "model.bin"
    if ((Test-Path $cpp) -and (Test-Path $bin)) {
        Write-Host "SKIP $($group.Name): existing output"
        continue
    }
    New-Item -ItemType Directory -Force $out | Out-Null
    $inputPath = (Resolve-Path $input).Path
    $outputPath = Join-Path (Resolve-Path $out).Path "model.cpp"
    Write-Host "CONVERT $($group.Name)"
    $logPath = Join-Path $out "conversion.log"
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($Debug) {
        & $python $converter --input_network $inputPath --output_path $outputPath --input_dim hidden_states 1,512,2560 --input_dim attention_mask 1,1,512,512 --input_dim cos 1,512,128 --input_dim sin 1,512,128 --no_simplification --debug *> $logPath
    } else {
        & $python $converter --input_network $inputPath --output_path $outputPath --input_dim hidden_states 1,512,2560 --input_dim attention_mask 1,1,512,512 --input_dim cos 1,512,128 --input_dim sin 1,512,128 --no_simplification *> $logPath
    }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    Get-Content $logPath -Tail 20
    if ($exitCode -ne 0) { throw "Conversion failed for $($group.Name), exit code $exitCode" }
    if (-not (Test-Path $cpp)) { throw "Conversion completed without CPP output for $($group.Name)" }
    if (-not (Test-Path $bin)) { throw "Conversion completed without BIN output for $($group.Name)" }
    Write-Host "DONE $($group.Name): $((Get-Item $bin).Length) bytes"
}
Write-Host "All Qwen3 layer groups converted under $OutputDir"
