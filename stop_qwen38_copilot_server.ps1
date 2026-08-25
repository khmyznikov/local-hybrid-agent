[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$windowsPidFile = Join-Path $PSScriptRoot ".qwen38-copilot-wsl.pid"
$stateFile = Join-Path $PSScriptRoot ".qwen38-copilot-profile.json"

if (Test-Path $windowsPidFile) {
    $windowsPid = [int](Get-Content $windowsPidFile -Raw)
    $windowsProcess = Get-Process -Id $windowsPid -ErrorAction SilentlyContinue
    if ($windowsProcess -and $windowsProcess.ProcessName -eq "wsl") {
        Stop-Process -Id $windowsPid -Force
    }
    Remove-Item $windowsPidFile -Force
}
Remove-Item $stateFile -Force -ErrorAction SilentlyContinue

$bashScript = @'
set -euo pipefail
PID_FILE=/home/gkhmyznikov/vllm-qwen38-wsl/qwen38_copilot_server.pid

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)
    if [[ "$CMD" == *"vllm"* && "$CMD" == *"Qwen3.8-27B-NVFP4-RTX5090"* ]]; then
        kill -TERM "$PID" || true
    fi
    rm -f "$PID_FILE"
fi

echo "Qwen server stopped"
'@
$bashScript = $bashScript.Replace("`r`n", "`n")
$encodedScript = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($bashScript)
)

& wsl.exe -d $Distribution -- bash -lc "echo $encodedScript | base64 -d | bash"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stop the Qwen server"
}