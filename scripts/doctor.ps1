[CmdletBinding()]
param(
    [ValidateSet("mock", "real")]
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

if (Test-Path -LiteralPath $avatarPython) {
    $avatarVersion = (& $avatarPython --version 2>&1 | Select-Object -First 1)
    $avatarOk = "$avatarVersion" -match "Python 3\.10\."
    Write-Check "Avatar Python" $(if ($avatarOk) { "PASS" } else { "FAIL" }) "$avatarVersion" ($ProviderMode -eq "real")
} else {
    Write-Check "Avatar Python" $(if ($ProviderMode -eq "real") { "FAIL" } else { "SKIP" }) "Python 3.10 worker environment is optional in mock mode" ($ProviderMode -eq "real")
}

$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
Write-Check "Node.js" $(if ($node) { "PASS" } else { "FAIL" }) $(if ($node) { (& $node.Source --version) } else { "node not found" })
Write-Check "npm" $(if ($npm) { "PASS" } else { "FAIL" }) $(if ($npm) { (& $npm.Source --version) } else { "npm.cmd not found" })

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
Write-Check "FFmpeg" $(if ($ffmpeg) { "PASS" } elseif ($ProviderMode -eq "real") { "FAIL" } else { "SKIP" }) $(if ($ffmpeg) { "available for Edge TTS PCM conversion" } else { "optional in mock mode" }) ($ProviderMode -eq "real")

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($ProviderMode -eq "real") {
    if ($docker) {
        & $docker.Source info *> $null
        Write-Check "Docker / LiveKit" $(if ($LASTEXITCODE -eq 0) { "PASS" } else { "FAIL" }) "Docker daemon must be ready for local LiveKit"
    } else {
        Write-Check "Docker / LiveKit" "FAIL" "docker not found"
    }
} else {
    Write-Check "Docker / LiveKit" "SKIP" "not required by the zero-GPU mock call"
}

if ($legacyRoot) {
    $personData = Join-Path $legacyRoot.Path "data\222"
    $headCheckpoint = Join-Path $legacyRoot.Path "trial_222\checkpoints\ngp.pth"
    $torsoCheckpoint = Join-Path $legacyRoot.Path "trial_222_torso\checkpoints\ngp.pth"
    $assetsReady = (Test-Path -LiteralPath $personData) -and
        (Test-Path -LiteralPath $headCheckpoint) -and
        (Test-Path -LiteralPath $torsoCheckpoint)
    Write-Check "Person 222 assets" $(if ($assetsReady) { "PASS" } else { "FAIL" }) "read-only data plus head/torso checkpoints" ($ProviderMode -eq "real")
} else {
    Write-Check "Person 222 assets" $(if ($ProviderMode -eq "real") { "FAIL" } else { "WARN" }) "legacy RAD-NeRF source not found" ($ProviderMode -eq "real")
}

$gpuTool = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($ProviderMode -eq "real") {
    Write-Check "CUDA GPU" $(if ($gpuTool) { "PASS" } else { "FAIL" }) $(if ($gpuTool) { "nvidia-smi is available; run the explicit GPU smoke separately" } else { "NVIDIA runtime not found" })
} else {
    Write-Check "CUDA GPU" "SKIP" "the approved baseline requires no GPU"
}

$qwenConfigured = -not [string]::IsNullOrWhiteSpace($env:PSYAVATAR_DASHSCOPE_API_KEY)
Write-Check "Qwen credentials" $(if ($ProviderMode -eq "mock") { "SKIP" } elseif ($qwenConfigured) { "PASS" } else { "FAIL" }) $(if ($ProviderMode -eq "mock") { "mock provider selected" } elseif ($qwenConfigured) { "credential is present" } else { "PSYAVATAR_DASHSCOPE_API_KEY is missing" }) ($ProviderMode -eq "real")

$listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners().Port
foreach ($port in 7880, 7881, 8000, 5173) {
    $inUse = $listeners -contains $port
    Write-Check "Port $port" $(if ($inUse) { "WARN" } else { "PASS" }) $(if ($inUse) { "already in use; this may be an existing demo service" } else { "available" }) $false
}

if ($requiredFailures -gt 0) {
    Write-Host "Diagnostics failed: $requiredFailures required check(s) did not pass."
    exit 1
}

Write-Host "Diagnostics passed for $ProviderMode mode."
