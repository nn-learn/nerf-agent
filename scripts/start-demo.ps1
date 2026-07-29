[CmdletBinding()]
param(
    [ValidateSet("mock", "real")]
    [string]$ProviderMode = "mock"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $projectRoot "runtime"
$orchestratorRoot = Join-Path $projectRoot "services\orchestrator"
$webRoot = Join-Path $projectRoot "apps\web"
$python = Join-Path $orchestratorRoot ".venv\Scripts\python.exe"
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$started = @()

function Start-HiddenDemoProcess {
    param(
        [string]$FilePath,
        [string]$Arguments,
        [string]$WorkingDirectory
    )
    # Windows PowerShell 5.1 can fail in Start-Process when the host supplies
    # duplicate Path/PATH environment keys. ShellExecute avoids rebuilding that
    # case-insensitive dictionary while preserving a hidden child window.
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $Arguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $true
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    return [System.Diagnostics.Process]::Start($startInfo)
}

function Wait-Http {
    param([string]$Url)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for $Url"
}

& (Join-Path $PSScriptRoot "doctor.ps1") -ProviderMode $ProviderMode
if ($LASTEXITCODE -ne 0) {
    throw "Diagnostics did not pass"
}
if (-not (Test-Path -LiteralPath (Join-Path $webRoot "node_modules"))) {
    throw "Frontend dependencies are missing. Run npm install in apps/web first."
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if ($ProviderMode -eq "real") {
    & docker compose -f (Join-Path $projectRoot "infra\docker-compose.yml") up -d
    if ($LASTEXITCODE -ne 0) {
        throw "LiveKit failed to start"
    }
}

$previousProviderMode = $env:PSYAVATAR_PROVIDER_MODE
$previousMediaMode = $env:VITE_MEDIA_MODE
try {
    $env:PSYAVATAR_PROVIDER_MODE = $ProviderMode
    $env:VITE_MEDIA_MODE = $(if ($ProviderMode -eq "real") { "livekit" } else { "mock" })
    $orchestrator = Start-HiddenDemoProcess `
        -FilePath $python `
        -Arguments "-m uvicorn app.main:app --host 127.0.0.1 --port 8000" `
        -WorkingDirectory $orchestratorRoot
    $started += $orchestrator
    $web = Start-HiddenDemoProcess `
        -FilePath $npm `
        -Arguments "run dev -- --host 127.0.0.1 --port 5173" `
        -WorkingDirectory $webRoot
    $started += $web
} finally {
    $env:PSYAVATAR_PROVIDER_MODE = $previousProviderMode
    $env:VITE_MEDIA_MODE = $previousMediaMode
}

try {
    Wait-Http "http://127.0.0.1:8000/health/ready"
    Wait-Http "http://127.0.0.1:5173/"
} catch {
    foreach ($process in $started) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    }
    throw
}

$started |
    Select-Object Id, ProcessName |
    ConvertTo-Json |
    Set-Content -Encoding utf8 (Join-Path $runtimeRoot "demo-processes.json")

Write-Host "PsyAvatar Care demo is ready."
Write-Host "User call:       http://127.0.0.1:5173/"
Write-Host "API docs:        http://127.0.0.1:8000/docs"
Write-Host "Clinician view:  http://127.0.0.1:5173/?view=clinician&session=<session_id>"
Write-Host "Process IDs are recorded in runtime/demo-processes.json."
