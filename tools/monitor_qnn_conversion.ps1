param(
    [string]$InputOnnx = "build\text_encoder_segments_v2\layers_00_05\model.onnx",
    [string]$OutputCpp = "build\text_encoder_segments_v2\qnn_layers_00_05.cpp",
    [int]$TimeoutSeconds = 900,
    [switch]$NoSimplification
)

$ErrorActionPreference = "Stop"
$qnn = if ($env:QNN_SDK_ROOT) { $env:QNN_SDK_ROOT } else { "E:\zimage-runtime\qairt\qairt\2.49.0.260730" }
$py = if ($env:QAIRT_PYTHON) { $env:QAIRT_PYTHON } else { "E:\zimage-runtime\python312\python.exe" }
$converter = Join-Path $qnn "bin\x86_64-windows-msvc\qnn-onnx-converter"
$env:PYTHONPATH = "$(Join-Path $qnn 'lib\python')"
# The converter is invoked by absolute path; avoid mutating PATH because some
# Windows environments expose both Path and PATH keys to Start-Process.
$input = (Resolve-Path $InputOnnx).Path
$output = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputCpp))
$outputDir = Split-Path -Parent $output
New-Item -ItemType Directory -Force $outputDir | Out-Null
$stdout = Join-Path $outputDir "conversion.stdout.log"
$stderr = Join-Path $outputDir "conversion.stderr.log"
Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue
$arguments = @(
    $converter,
    "--input_network", $input,
    "--output_path", $output,
    "--input_dim", "hidden_states", "1,512,2560",
    "--input_dim", "attention_mask", "1,1,512,512",
    "--input_dim", "cos", "1,512,128",
    "--input_dim", "sin", "1,512,128",
    "--debug"
)
if ($NoSimplification) { $arguments += "--no_simplification" }
$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = $py
$info.UseShellExecute = $false
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
$info.CreateNoWindow = $true
$info.Arguments = ($arguments | ForEach-Object { if ($_ -match '[\s\"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $info
[void]$process.Start()
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
Write-Output "PID=$($process.Id)"
$started = Get-Date
while (-not $process.HasExited) {
    $process.Refresh()
    $available = $null
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $available = [math]::Round($os.FreePhysicalMemory / 1024)
    } catch {
        $available = "unknown"
    }
    $current = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
    if ($current) {
        Write-Output ((Get-Date).ToString("HH:mm:ss") + " elapsed=" + [math]::Round(((Get-Date) - $started).TotalSeconds) + "s wsMB=" + [math]::Round($current.WorkingSet64 / 1MB) + " privateMB=" + [math]::Round($current.PrivateMemorySize64 / 1MB) + " cpu=" + [math]::Round($current.CPU, 1) + " runtimeAvailableMB=" + $available)
    }
    if (((Get-Date) - $started).TotalSeconds -gt $TimeoutSeconds) {
        Write-Output "TIMEOUT=$TimeoutSeconds"
        Stop-Process -Id $process.Id -Force
        break
    }
    Start-Sleep -Seconds 5
}
$process.Refresh()
$stdoutTask.Result | Set-Content $stdout
$stderrTask.Result | Set-Content $stderr
Write-Output "EXIT=$($process.ExitCode)"
Get-Content $stdout -Tail 80
Get-Content $stderr -Tail 120
