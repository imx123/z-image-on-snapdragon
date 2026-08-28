param(
    [string]$QnnSdkRoot = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "zimage-runtime\qairt\qairt\2.49.0.260730"),
    [string]$NdkRoot = "C:\Users\Max\AppData\Local\Android\Sdk\ndk\28.0.13004108",
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\qnn_transformer_segments_w4a8"),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\android\jniLibs\arm64-v8a"),
    [string[]]$Segments = @('frontend','layers_00_05','layers_06_11','layers_12_17','layers_18_23','layers_24_29','final'),
    [switch]$ForceRebuild
)

$ErrorActionPreference = 'Stop'
$ndkBuild = Join-Path $NdkRoot 'ndk-build.cmd'
$objcopy = Join-Path $NdkRoot 'toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-objcopy.exe'
foreach ($path in @($QnnSdkRoot, $ndkBuild, $objcopy)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required Android/QNN path not found: $path" }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$templateRoot = Join-Path $QnnSdkRoot 'share\QNN\converter\jni'
foreach ($segment in $Segments) {
    $inputDir = Join-Path $InputRoot $segment
    # ndk-build canonicalizes 8.3 aliases, so keep the temporary project outside
    # the workspace path whose name contains spaces.
    $workDir = "E:\qnn_android_build\$segment"
    $outLib = Join-Path $OutputRoot "libqnn_transformer_$segment.so"
    if ((Test-Path -LiteralPath $outLib) -and -not $ForceRebuild) { Write-Host "[skip] $outLib"; continue }
    if (-not (Test-Path (Join-Path $inputDir 'model.cpp')) -or -not (Test-Path (Join-Path $inputDir 'model.bin'))) { throw "Incomplete QNN input: $inputDir" }
    if (Test-Path $workDir) { Remove-Item -LiteralPath $workDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $workDir | Out-Null
    Copy-Item (Join-Path $inputDir 'model.cpp') $workDir
    Copy-Item (Join-Path $inputDir 'model.bin') $workDir
    foreach ($supportFile in @('QnnModel.cpp','QnnModel.hpp','QnnModelPal.hpp','QnnTypeMacros.hpp','QnnWrapperUtils.cpp','QnnWrapperUtils.hpp')) {
        Copy-Item (Join-Path $templateRoot $supportFile) $workDir
    }
    # strnDup() lives in the linux/ platform implementation and is required by
    # QnnModel.cpp/QnnWrapperUtils.cpp. The old model libs that loaded on the
    # device all shipped this translation unit as well.
    Copy-Item (Join-Path $templateRoot 'linux\QnnModelPal.cpp') (Join-Path $workDir 'QnnModelPal.cpp')
    $binaryDir = Join-Path $workDir 'obj\binary'
    New-Item -ItemType Directory -Force -Path $binaryDir | Out-Null
    # tar.exe on Windows treats "E:\..." as a remote host; use Python's tarfile
    # module instead (handles absolute Windows paths reliably).
    $pyExtract = @"
import sys, tarfile, pathlib
src, dst = sys.argv[1], sys.argv[2]
pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
with tarfile.open(src, 'r:') as t:
    t.extractall(dst)
print('extracted', src)
"@
    $pyExe = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "E:\projects\zimage on phone\zimage-runtime\python312\python.exe" }
    & $pyExe -c $pyExtract (Join-Path $inputDir 'model.bin') $binaryDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract model.bin for $segment" }
    $androidMk = (Get-Content -Raw (Join-Path $templateRoot 'Android.mk'))
    $androidMk = $androidMk -replace 'QNN_SDK_ROOT := $(abspath (abspath $(lastword $(LOCAL_PATH)))/../../../../..)', "QNN_SDK_ROOT := $($QnnSdkRoot -replace '\\','/')"
    $androidMk = $androidMk -replace '$(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objcopy', "$($objcopy -replace '\\','/')"
    $androidMk = $androidMk.Replace('$(shell mkdir -p $(BINARY_DIR) $(BINARY_OBJ_DIR))', '')
    $androidMk = $androidMk.Replace('$(shell tar xf $(BINARY_FILE) -C $(BINARY_DIR) >/dev/null)', '')
    $androidMk = $androidMk.Replace('prebuilt/linux-x86_64/bin/llvm-objcopy', 'prebuilt/windows-x86_64/bin/llvm-objcopy')
    # objcopy must see the relative path obj/binary/<file>.raw; an absolute
    # Windows path embeds E__qnn_android_build_<seg>_obj_binary_... in the
    # symbol names and breaks BINVARSTART/BINLEN references.
    $androidMk = $androidMk.Replace('$(NDK_OBJCOPY_CMD) $< $@', 'cd $(LOCAL_PATH) && $(NDK_OBJCOPY_CMD) obj/binary/$(notdir $*).raw $@')
    Set-Content -LiteralPath (Join-Path $workDir 'Android.mk') -Value $androidMk -NoNewline
    $appMk = Get-Content -Raw (Join-Path $templateRoot 'Application.mk')
    Set-Content -LiteralPath (Join-Path $workDir 'Application.mk') -Value $appMk -NoNewline
    Push-Location $workDir
    try {
        $env:QNN_SDK_ROOT = ($QnnSdkRoot -replace '\\','/')
        $env:QNN_ANDROID_APP_ABIS = 'arm64-v8a'
        & $ndkBuild V=1 NDK_PROJECT_PATH=$workDir APP_BUILD_SCRIPT=$workDir/Android.mk NDK_APPLICATION_MK=$workDir/Application.mk
        if ($LASTEXITCODE -ne 0) { throw "ndk-build failed for $segment ($LASTEXITCODE)" }
    } finally { Pop-Location }
    $built = Join-Path $workDir "libs\arm64-v8a\libmodel.so"
    if (-not (Test-Path $built)) { throw "Expected output missing: $built" }
    Copy-Item $built $outLib
    Write-Host "[built] $outLib ($((Get-Item $outLib).Length) bytes)"
}
