param([string]$QnnSdkRoot = $(if ($env:QNN_SDK_ROOT) { $env:QNN_SDK_ROOT } else { (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "zimage-runtime\qairt\qairt\2.49.0.260730") }))
if (-not (Test-Path (Join-Path $QnnSdkRoot "bin\x86_64-windows-msvc\qnn-onnx-converter"))) { throw "Invalid QAIRT root: $QnnSdkRoot" }
$env:QNN_SDK_ROOT = (Resolve-Path $QnnSdkRoot).Path
$env:Path = "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc;$env:Path"
Write-Host "QNN_SDK_ROOT=$env:QNN_SDK_ROOT"
Get-Item "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc\qnn-onnx-converter","$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc\qnn-model-lib-generator","$env:QNN_SDK_ROOT\bin\aarch64-android\qnn-context-binary-generator" | Select-Object FullName,Length
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python;$env:PYTHONPATH"
$env:QAIRT_PYTHON = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "zimage-runtime\python312\python.exe") }
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { (Join-Path ([System.IO.Path]::GetTempPath()) "zimage-qairt") }
New-Item -ItemType Directory -Force $env:QAIRT_TMP_DIR | Out-Null
