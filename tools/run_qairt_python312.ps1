param([Parameter(Mandatory=$true)][string]$Script, [string[]]$Arguments)
$qnn = if ($env:QNN_SDK_ROOT) { $env:QNN_SDK_ROOT } else { "C:\Qualcomm\qairt\qairt\2.49.0.260730" }
$py = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "C:\Qualcomm\qairt\runtime\python312\python.exe" }
if (-not (Test-Path $py)) { throw "QAIRT Python runtime not found: $py" }
if (-not (Test-Path (Join-Path $qnn "lib\python\qti"))) { throw "Invalid QNN_SDK_ROOT: $qnn" }
$env:QNN_SDK_ROOT = (Resolve-Path $qnn).Path
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python;$env:PYTHONPATH"
$env:PATH = "$env:QNN_SDK_ROOT\lib\python\qti\aisw\converters\common\windows-x86_64;$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc;$env:PATH"
$env:QAIRT_TMP_DIR = if ($env:QAIRT_TMP_DIR) { $env:QAIRT_TMP_DIR } else { (Join-Path ([System.IO.Path]::GetTempPath()) "zimage-qairt") }
New-Item -ItemType Directory -Force $env:QAIRT_TMP_DIR | Out-Null
& $py $Script @Arguments
exit $LASTEXITCODE
