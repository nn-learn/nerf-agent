[CmdletBinding()]
param(
    [ValidateSet("mock", "local", "real")]
    [string]$ProviderMode = "mock"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$orchestratorPython = Join-Path $projectRoot "services\orchestrator\.venv\Scripts\python.exe"
$avatarPython = Join-Path $projectRoot "services\avatar_radnerf\.venv\Scripts\python.exe"
$legacyRoot = (Resolve-Path (Join-Path $projectRoot "..\RAD-NeRF\RAD-NeRF") -ErrorAction SilentlyContinue)
$requiredFailures = 0

function Write-Check {
    param(
        [string]$Name,
        [ValidateSet("PASS", "WARN", "SKIP", "FAIL")]
        [string]$Status,
        [string]$Detail,
        [bool]$Required = $true
    )
    if ($Status -eq "FAIL" -and $Required) {
        $script:requiredFailures += 1
    }
    Write-Host ("[{0,-4}] {1} - {2}" -f $Status, $Name, $Detail)
}

Write-Host "PsyAvatar Care diagnostics ($ProviderMode)"
Write-Host "Secrets are checked for presence only; values are never printed."

if (Test-Path -LiteralPath $orchestratorPython) {
    $pythonVersion = (& $orchestratorPython --version 2>&1 | Select-Object -First 1)
    $pythonOk = "$pythonVersion" -match "Python 3\.12\."
    Write-Check "Orchestrator Python" $(if ($pythonOk) { "PASS" } else { "FAIL" }) "$pythonVersion"
} else {
    Write-Check "Orchestrator Python" "FAIL" "services/orchestrator/.venv is missing"
}

if ($ProviderMode -eq "real") {
    if (Test-Path -LiteralPath $avatarPython) {
        $avatarVersion = (& $avatarPython --version 2>&1 | Select-Object -First 1)
        $avatarOk = "$avatarVersion" -match "Python 3\.10\."
        Write-Check "Avatar Python" $(if ($avatarOk) { "PASS" } else { "FAIL" }) "$avatarVersion"
    } else {
        Write-Check "Avatar Python" "FAIL" "Python 3.10 worker environment is missing"
    }
} else {
    Write-Check "Avatar Python" "SKIP" "not used by $ProviderMode mode"
}

$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
Write-Check "Node.js" $(if ($node) { "PASS" } else { "FAIL" }) $(if ($node) { (& $node.Source --version) } else { "node not found" })
Write-Check "npm" $(if ($npm) { "PASS" } else { "FAIL" }) $(if ($npm) { (& $npm.Source --version) } else { "npm.cmd not found" })

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffmpegRequired = $ProviderMode -in @("local", "real")
Write-Check "FFmpeg" $(if ($ffmpeg) { "PASS" } elseif ($ffmpegRequired) { "FAIL" } else { "SKIP" }) $(if ($ffmpeg) { "available for Edge TTS PCM conversion" } elseif ($ffmpegRequired) { "required for Edge TTS PCM conversion" } else { "not used by mock mode" }) $ffmpegRequired

if ($ProviderMode -eq "real") {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        & $docker.Source info *> $null
        Write-Check "Docker / LiveKit" $(if ($LASTEXITCODE -eq 0) { "PASS" } else { "FAIL" }) "Docker daemon must be ready for local LiveKit"
    } else {
        Write-Check "Docker / LiveKit" "FAIL" "docker not found"
    }
} else {
    Write-Check "Docker / LiveKit" "SKIP" "not used by $ProviderMode mode"
}

if ($ProviderMode -eq "real") {
    if ($legacyRoot) {
        $personData = Join-Path $legacyRoot.Path "data\222"
        $headCheckpoint = Join-Path $legacyRoot.Path "trial_222\checkpoints\ngp.pth"
        $torsoCheckpoint = Join-Path $legacyRoot.Path "trial_222_torso\checkpoints\ngp.pth"
        $assetsReady = (Test-Path -LiteralPath $personData) -and
            (Test-Path -LiteralPath $headCheckpoint) -and
            (Test-Path -LiteralPath $torsoCheckpoint)
        Write-Check "Person 222 assets" $(if ($assetsReady) { "PASS" } else { "FAIL" }) "read-only data plus head/torso checkpoints"
    } else {
        Write-Check "Person 222 assets" "FAIL" "legacy RAD-NeRF source not found"
    }
} else {
    Write-Check "Person 222 assets" "SKIP" "not used by $ProviderMode mode"
}

if ($ProviderMode -eq "real") {
    $gpuTool = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    Write-Check "CUDA GPU" $(if ($gpuTool) { "PASS" } else { "FAIL" }) $(if ($gpuTool) { "nvidia-smi is available; run the explicit GPU smoke separately" } else { "NVIDIA runtime not found" })
} else {
    Write-Check "CUDA GPU" "SKIP" "not used by $ProviderMode mode"
}

if ($ProviderMode -eq "real") {
    $qwenConfigured = -not [string]::IsNullOrWhiteSpace($env:PSYAVATAR_DASHSCOPE_API_KEY)
    Write-Check "Qwen credentials" $(if ($qwenConfigured) { "PASS" } else { "FAIL" }) $(if ($qwenConfigured) { "credential is present" } else { "PSYAVATAR_DASHSCOPE_API_KEY is missing" })
} else {
    Write-Check "Qwen credentials" "SKIP" "not used by $ProviderMode mode"
}

if ($ProviderMode -eq "local") {
    $ollamaBaseUrl = $env:PSYAVATAR_OLLAMA_BASE_URL
    if ([string]::IsNullOrWhiteSpace($ollamaBaseUrl)) {
        $ollamaBaseUrl = "http://127.0.0.1:11434"
    }
    $ollamaBaseUrl = $ollamaBaseUrl.TrimEnd("/")
    $textModel = $env:PSYAVATAR_TEXT_MODEL
    if ([string]::IsNullOrWhiteSpace($textModel)) {
        $textModel = "qwen3.6:latest"
    }

    try {
        $ollamaVersion = Invoke-RestMethod -Uri "$ollamaBaseUrl/api/version" -TimeoutSec 5
        $versionValue = [string]$ollamaVersion.version
        Write-Check "Ollama version" $(if ([string]::IsNullOrWhiteSpace($versionValue)) { "FAIL" } else { "PASS" }) $(if ([string]::IsNullOrWhiteSpace($versionValue)) { "version response is invalid" } else { "local service responded" })
    } catch {
        Write-Check "Ollama version" "FAIL" "local service is unavailable"
    }

    try {
        $ollamaTags = Invoke-RestMethod -Uri "$ollamaBaseUrl/api/tags" -TimeoutSec 10
        $modelInstalled = @($ollamaTags.models | Where-Object { $_.name -eq $textModel }).Count -gt 0
        Write-Check "Ollama model" $(if ($modelInstalled) { "PASS" } else { "FAIL" }) $(if ($modelInstalled) { "exact model installed: $textModel" } else { "exact model missing: $textModel" })
    } catch {
        Write-Check "Ollama model" "FAIL" "could not query installed models"
    }

    if (Test-Path -LiteralPath $orchestratorPython) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "SilentlyContinue"
            & $orchestratorPython -c "import edge_tts, faster_whisper, sentence_transformers, webrtcvad" *> $null
            $importsReady = $LASTEXITCODE -eq 0
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        Write-Check "Speech / RAG imports" $(if ($importsReady) { "PASS" } else { "FAIL" }) $(if ($importsReady) { "faster-whisper, Edge TTS, WebRTC VAD and sentence-transformers import successfully" } else { "run scripts/bootstrap.ps1 to install the local extras" })
    } else {
        Write-Check "Speech / RAG imports" "FAIL" "orchestrator Python is unavailable"
    }
} else {
    Write-Check "Ollama" "SKIP" "not used by $ProviderMode mode"
    Write-Check "Speech / RAG imports" "SKIP" "not used by $ProviderMode mode"
}

$listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners().Port
$ports = if ($ProviderMode -eq "real") { 7880, 7881, 8000, 5173 } else { 8000, 5173 }
foreach ($port in $ports) {
    $inUse = $listeners -contains $port
    Write-Check "Port $port" $(if ($inUse) { "WARN" } else { "PASS" }) $(if ($inUse) { "already in use; this may be an existing demo service" } else { "available" }) $false
}

if ($requiredFailures -gt 0) {
    Write-Host "Diagnostics failed: $requiredFailures required check(s) did not pass."
    exit 1
}

Write-Host "Diagnostics passed for $ProviderMode mode."
