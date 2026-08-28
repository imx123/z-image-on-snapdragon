param(
    [string]$Adb = "adb",
    [string]$Device = "",
    [string]$RuntimeRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$RemoteRoot = "/storage/emulated/0/Android/data/com.example.zimage/files/zimage-runtime",
    [string]$VaeLib = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\android\jniLibs\arm64-v8a\libqnn_vae_shiftfix.so"),
    [string]$TransformerLibRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\android\jniLibs-transformer-v6-fp16"),
    [string]$TextEncoderLibRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\android\jniLibs-text-fp16"),
    [string]$FreqsAsset = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\transformer_freqs_qnn\unified_freqs_f32_nhwc.raw"),
    [string]$TextPrecompRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\text_encoder_precomp"),
    [string]$QnnLibRoot = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "zimage-runtime\qairt\qairt\2.49.0.260730\lib\aarch64-android"),
    [switch]$IncludeModel
)

$ErrorActionPreference = "Stop"
$adbArgs = @()
if ($Device) { $adbArgs += @("-s", $Device) }
function Invoke-Adb([string[]]$CommandArgs) {
    $allArgs = @($adbArgs) + @($CommandArgs)
    & $Adb @allArgs
    if ($LASTEXITCODE -ne 0) { throw "adb failed: $($allArgs -join ' ')" }
}
function Sync-File([string]$LocalPath, [string]$RemotePath, [switch]$Force) {
    $file = Get-Item -LiteralPath $LocalPath
    if (-not $Force) {
        $previous = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        $raw = & $Adb @($adbArgs + @("shell", "stat", "-c", "%s", $RemotePath)) 2>$null
        $size = if ($null -eq $raw) { "" } else { ($raw -join "").Trim() }
        $ErrorActionPreference = $previous
        if ($size -eq $file.Length.ToString()) { Write-Host "SKIP $($file.Name)"; return }
    }
    Invoke-Adb @("push", $file.FullName, $RemotePath)
}

# Keep the shared-storage copy persistent and remove only the obsolete staging path.
Invoke-Adb @("shell", "rm", "-rf", "/data/local/tmp/zimage-runtime")
Invoke-Adb @("shell", "mkdir", "-p", "$RemoteRoot/lib/arm64-v8a", "$RemoteRoot/assets", "$RemoteRoot/models")

# FP16 v6 model libraries + corrected VAE. Always force-push these: the size-only
# skip check is not safe when a rebuilt .so has the same byte length.
Sync-File $VaeLib "$RemoteRoot/lib/arm64-v8a/libqnn_vae_shiftfix.so" -Force
Get-ChildItem (Join-Path $TransformerLibRoot "arm64-v8a") -Filter "libqnn_transformer_*.so" | ForEach-Object {
    Sync-File $_.FullName "$RemoteRoot/lib/arm64-v8a/$($_.Name)" -Force
}
Get-ChildItem (Join-Path $TextEncoderLibRoot "arm64-v8a") -Filter "*.so" | ForEach-Object {
    Sync-File $_.FullName "$RemoteRoot/lib/arm64-v8a/$($_.Name)" -Force
}
Sync-File $FreqsAsset "$RemoteRoot/assets/unified_freqs_f32_nhwc.raw" -Force
Sync-File (Join-Path $TextPrecompRoot "cos_qnn_f32.raw") "$RemoteRoot/assets/cos_qnn_f32.raw" -Force
Sync-File (Join-Path $TextPrecompRoot "sin_qnn_f32.raw") "$RemoteRoot/assets/sin_qnn_f32.raw" -Force
Sync-File (Join-Path $TextPrecompRoot "causal_mask_qnn_f32.raw") "$RemoteRoot/assets/causal_mask_qnn_f32.raw" -Force

$qnnLibDir = $QnnLibRoot
if (-not (Test-Path $qnnLibDir)) { throw "QNN Android library root not found: $qnnLibDir" }
Get-ChildItem $qnnLibDir -Filter "*.so" | ForEach-Object {
    Sync-File $_.FullName "$RemoteRoot/lib/arm64-v8a/$($_.Name)"
}
# Hexagon skel libraries are REQUIRED for HTP transport in the app sandbox
# (loaded via ADSP_LIBRARY_PATH). CRITICAL: deploy_android_runtime must push
# them or HTP deviceCreate fails with 14001 ("Unable to load lib"). See
# PROJECT_HANDOFF_ARCHIVE.md "Solution Update 2026-08-16".
$hexDir = Join-Path $QnnLibRoot "..\..\lib\hexagon-v79\unsigned"
if (Test-Path $hexDir) {
    Get-ChildItem $hexDir -Filter "*HtpV79Skel.so" | ForEach-Object {
        Sync-File $_.FullName "$RemoteRoot/lib/arm64-v8a/$($_.Name)" -Force
    }
} else { Write-Host "WARN: hexagon-v79/unsigned skel dir not found: $hexDir" }

# Public-readable permissions are required: adb-pushed files can otherwise be
# shell:ext_data_rw 770 and invisible to the app uid.
Invoke-Adb @("shell", "chmod", "-R", "a+rX", $RemoteRoot)

if ($IncludeModel) {
    $model = Join-Path $RuntimeRoot "models\z-image-turbo\models\Tongyi-MAI--Z-Image-Turbo\snapshots\master"
    if (-not (Test-Path $model)) { throw "Model root not found: $model" }
    Invoke-Adb @("push", $model, "$RemoteRoot/models/")
}
Write-Host "Android runtime FP16 v6 deployed under $RemoteRoot (override with -TransformerLibRoot / -TextEncoderLibRoot)"
