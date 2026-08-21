$ErrorActionPreference = "Stop"
$ServerPort = 8001
$CopilotArgs = [Collections.Generic.List[string]]::new()
for ($index = 0; $index -lt $args.Count; $index++) {
    if ($args[$index] -eq "-ServerPort") {
        if ($index + 1 -ge $args.Count) {
            throw "-ServerPort requires a value"
        }
        $ServerPort = [int]$args[++$index]
    } else {
        $CopilotArgs.Add([string]$args[$index])
    }
}

& "$PSScriptRoot\start_qwen38_copilot_server.ps1" -Port $ServerPort

$env:COPILOT_PROVIDER_BASE_URL = "http://127.0.0.1:$ServerPort/v1"
$env:COPILOT_PROVIDER_TYPE = "openai"
$env:COPILOT_PROVIDER_API_KEY = "local-copilot"
$env:COPILOT_PROVIDER_WIRE_API = "completions"
$env:COPILOT_MODEL = "qwen3.8-27b-local"
$env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS = "240000"
$env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = "8192"

& copilot @($CopilotArgs)
exit $LASTEXITCODE