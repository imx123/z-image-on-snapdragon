param(
    [string]$Adb = "adb",
    [string]$Device = "",
    [string]$QnnSdkRoot = "E:\qnn",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$FrontendLib = (Join-Path $ProjectRoot "build\android\jniLibs-transformer-v9-fp32-test\arm64-v8a\libqnn_transformer_frontend.so"),
    [string]$InputDir = (Join-Path $ProjectRoot "build\probe_inputs_v9_fp32"),
    [string]$RemoteDir = "/data/local/tmp/gputest",
    [ValidateSet("gpu","htp","cpu")] [string]$Backend = "gpu"
)
# Offline-ready device test for the transformer frontend only.
# Device disconnected on 2026-08-18; run this once adb devices shows the phone.
$ErrorActionPreference = "Stop"
$adbArgs = @()
if ($Device) { $adbArgs += @("-s", $Device) }
function Invoke-Adb([string[]]$CommandArgs) {
    $allArgs = @($adbArgs) + @($CommandArgs)
    & $Adb @allArgs
    if ($LASTEXITCODE -ne 0) { throw "adb failed: $($allArgs -join ' ')" }
}
function Sync-Push([string]$Local, [string]$Remote) {
    if (-not (Test-Path -LiteralPath $Local)) { throw "Missing: $Local" }
    Invoke-Adb @("push", $Local, $Remote)
}

Invoke-Adb @("shell", "rm", "-rf", $RemoteDir)
Invoke-Adb @("shell", "mkdir", "-p", "$RemoteDir/inputs", "$RemoteDir/out")

$qnnBin = Join-Path $QnnSdkRoot "bin\aarch64-android\qnn-net-run"
Sync-Push $qnnBin "$RemoteDir/qnn-net-run"
if ($Backend -eq "gpu") {
    Sync-Push (Join-Path $QnnSdkRoot "lib\aarch64-android\libQnnGpu.so") "$RemoteDir/libQnnGpu.so"
} elseif ($Backend -eq "htp") {
    Sync-Push (Join-Path $QnnSdkRoot "lib\aarch64-android\libQnnHtp.so") "$RemoteDir/libQnnHtp.so"
    Sync-Push (Join-Path $QnnSdkRoot "lib\aarch64-android\libQnnHtpV79Stub.so") "$RemoteDir/libQnnHtpV79Stub.so"
    Sync-Push (Join-Path $QnnSdkRoot "lib\aarch64-android\libQnnHtpPrepare.so") "$RemoteDir/libQnnHtpPrepare.so"
    Sync-Push (Join-Path $QnnSdkRoot "lib\hexagon-v79\unsigned\libQnnHtpV79Skel.so") "$RemoteDir/libQnnHtpV79Skel.so"
} else {
    Sync-Push (Join-Path $QnnSdkRoot "lib\aarch64-android\libQnnCpu.so") "$RemoteDir/libQnnCpu.so"
}
Sync-Push $FrontendLib "$RemoteDir/libqnn_transformer_frontend.so"
Get-ChildItem (Join-Path $InputDir "*") -File | ForEach-Object { Sync-Push $_.FullName "$RemoteDir/inputs/$($_.Name)" }

$backendSo = switch ($Backend) { "gpu" { "libQnnGpu.so" } "htp" { "libQnnHtp.so" } "cpu" { "libQnnCpu.so" } }
$extraEnv = if ($Backend -eq "htp") { "export ADSP_LIBRARY_PATH=$RemoteDir &&" } else { "" }
$cmd = "cd $RemoteDir && export LD_LIBRARY_PATH=$RemoteDir && $extraEnv chmod 755 qnn-net-run && ./qnn-net-run --model libqnn_transformer_frontend.so --backend $backendSo --input_list inputs/input_list.txt --output_dir out_$Backend --use_native_input_files --use_native_output_files --log_level info"
Write-Host $cmd
Invoke-Adb @("shell", $cmd)
Write-Host "Outputs:" 
Invoke-Adb @("shell", "ls", "-la", "$RemoteDir/out_$Backend/Result_0")
