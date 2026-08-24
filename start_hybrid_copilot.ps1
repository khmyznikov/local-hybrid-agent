$ErrorActionPreference = "Stop"
$ServerPort = 8001
$ServerProfile = "Quality"
$CopilotArgs = [Collections.Generic.List[string]]::new()
for ($index = 0; $index -lt $args.Count; $index++) {
    if ($args[$index] -eq "-ServerPort") {
        if ($index + 1 -ge $args.Count) {
            throw "-ServerPort requires a value"
        }
        $ServerPort = [int]$args[++$index]
    } elseif ($args[$index] -eq "-ServerProfile") {
        if ($index + 1 -ge $args.Count) {
            throw "-ServerProfile requires Quality or Capacity"
        }
        $ServerProfile = [string]$args[++$index]
        if ($ServerProfile -notin @("Quality", "Capacity")) {
            throw "-ServerProfile requires Quality or Capacity"
        }
    } else {
        $CopilotArgs.Add([string]$args[$index])
    }
}

& "$PSScriptRoot\start_qwen38_copilot_server.ps1" `
    -Port $ServerPort -Profile $ServerProfile
$env:LOCAL_SIDEKICK_BASE_URL = "http://127.0.0.1:$ServerPort/v1"
$env:LOCAL_SIDEKICK_API_KEY = "local-copilot"
$env:LOCAL_SIDEKICK_MODEL = "qwen3.8-27b-local"
$env:LOCAL_SIDEKICK_TIMEOUT_SECONDS = "300"
$env:LOCAL_SIDEKICK_MAX_CONTEXT_CHARS = "60000"
$env:LOCAL_SIDEKICK_PROFILE = $ServerProfile.ToLowerInvariant()
if ($ServerProfile -eq "Quality") {
    $env:LOCAL_SIDEKICK_SHARED_CACHE_TOKENS = "175529"
    $env:LOCAL_SIDEKICK_NATIVE_MAX_REQUEST_TOKENS = "131072"
    $env:LOCAL_SIDEKICK_MAX_PROMPT_TOKENS = "120000"
} else {
    $env:LOCAL_SIDEKICK_SHARED_CACHE_TOKENS = "351058"
    $env:LOCAL_SIDEKICK_NATIVE_MAX_REQUEST_TOKENS = "262144"
    $env:LOCAL_SIDEKICK_MAX_PROMPT_TOKENS = "240000"
}

foreach ($name in @(
    "COPILOT_PROVIDER_BASE_URL",
    "COPILOT_PROVIDER_TYPE",
    "COPILOT_PROVIDER_API_KEY",
    "COPILOT_PROVIDER_BEARER_TOKEN",
    "COPILOT_PROVIDER_WIRE_API",
    "COPILOT_PROVIDER_TRANSPORT",
    "COPILOT_PROVIDER_MODEL_ID",
    "COPILOT_PROVIDER_WIRE_MODEL",
    "COPILOT_PROVIDER_MAX_PROMPT_TOKENS",
    "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS",
    "COPILOT_MODEL"
)) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$mcpConfig = "@$PSScriptRoot\copilot-local-sidekick.mcp.json"
& copilot --additional-mcp-config $mcpConfig `
    --allow-all-mcp-server-instructions `
    --allow-tool=local-qwen-sidekick @($CopilotArgs)
exit $LASTEXITCODE