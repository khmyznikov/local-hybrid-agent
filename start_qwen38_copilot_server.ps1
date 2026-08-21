[CmdletBinding()]
param(
    [int]$Port = 8001,
    [ValidateRange(1, 4)]
    [int]$MaxNumSeqs = 4,
    [string]$Distribution = "Ubuntu",
    [int]$WaitSeconds = 180
)

$ErrorActionPreference = "Stop"
$baseUrl = "http://127.0.0.1:$Port"
$windowsPidFile = Join-Path $PSScriptRoot ".qwen38-copilot-wsl.pid"

try {
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -TimeoutSec 2 | Out-Null
    Write-Host "Qwen server is already healthy at $baseUrl/v1"
    return
} catch {
}

if (Test-Path $windowsPidFile) {
    $oldWindowsPid = [int](Get-Content $windowsPidFile -Raw)
    if (Get-Process -Id $oldWindowsPid -ErrorAction SilentlyContinue) {
        Stop-Process -Id $oldWindowsPid -Force
    }
    Remove-Item $windowsPidFile -Force
}

$bashScript = @'
set -euo pipefail

PID_FILE=/home/gkhmyznikov/vllm-qwen38-wsl/qwen38_copilot_server.pid
LOG_FILE=/home/gkhmyznikov/vllm-qwen38-wsl/qwen38_copilot_server.log

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        kill -TERM "$OLD_PID" || true
    fi
    rm -f "$PID_FILE"
fi

: > "$LOG_FILE"
echo $$ > "$PID_FILE"
cd /tmp
exec env \
    CUDA_HOME=/usr/local/cuda \
    HOME=/home/gkhmyznikov \
    PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    MAX_JOBS=1 \
    /home/gkhmyznikov/vllm-qwen38-wsl/.venv/bin/vllm serve \
    /home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-RTX5090 \
    --host 0.0.0.0 \
    --port __PORT__ \
    --api-key local-copilot \
    --served-model-name qwen3.8-27b-local \
    --load-format safetensors \
    --safetensors-load-strategy lazy \
    --language-model-only \
    --max-model-len 262144 \
    --max-num-seqs __MAX_NUM_SEQS__ \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.01 \
    --kv-cache-memory-bytes 11811160064 \
    --kv-cache-dtype fp8 \
    --disable-custom-all-reduce \
    --gdn-prefill-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --kernel-config '{"linear_backend":"auto","enable_flashinfer_autotune":false,"enable_cutedsl_warmup":false,"enable_jit_warmup":false}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
    >>"$LOG_FILE" 2>&1
'@
$bashScript = $bashScript.Replace("__PORT__", $Port.ToString())
$bashScript = $bashScript.Replace("__MAX_NUM_SEQS__", $MaxNumSeqs.ToString())
$bashScript = $bashScript.Replace("`r`n", "`n")
$encodedScript = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($bashScript)
)
$bashCommand = "echo $encodedScript | base64 -d | bash"
$wslProcess = Start-Process -FilePath "$env:WINDIR\System32\wsl.exe" `
    -ArgumentList @(
        "-d",
        $Distribution,
        "--",
        "bash",
        "-lc",
        "`"$bashCommand`""
    ) `
    -WindowStyle Hidden `
    -PassThru
$wslProcess.Id | Set-Content -Path $windowsPidFile -NoNewline
Write-Host "Started WSL host PID $($wslProcess.Id); log: /home/gkhmyznikov/vllm-qwen38-wsl/qwen38_copilot_server.log"

$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
do {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -TimeoutSec 2 | Out-Null
        Write-Host "Qwen OpenAI endpoint is ready at $baseUrl/v1"
        Write-Host "Model: qwen3.8-27b-local"
        return
    } catch {
        if ($wslProcess.HasExited) {
            break
        }
        [Threading.Thread]::Sleep(2000)
    }
} while ([DateTime]::UtcNow -lt $deadline)

& wsl.exe -d $Distribution -- bash -lc "tail -80 /home/gkhmyznikov/vllm-qwen38-wsl/qwen38_copilot_server.log"
if (-not $wslProcess.HasExited) {
    Stop-Process -Id $wslProcess.Id -Force
}
Remove-Item $windowsPidFile -Force -ErrorAction SilentlyContinue
throw "Qwen server did not become healthy within $WaitSeconds seconds"