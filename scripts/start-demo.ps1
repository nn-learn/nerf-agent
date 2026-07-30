[CmdletBinding()]
param(
    [ValidateSet("mock", "local", "real")]
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

function Get-HttpStatusCode {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)
    $response = $ErrorRecord.Exception.Response
    if ($null -eq $response) {
        return $null
    }
    try {
        return [int]$response.StatusCode
    } catch {
        return $null
    }
}

function Get-HttpErrorPayload {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)
    $body = $ErrorRecord.ErrorDetails.Message
    if (-not [string]::IsNullOrWhiteSpace($body)) {
        try {
            return $body | ConvertFrom-Json
        } catch {
        }
    }

    $response = $ErrorRecord.Exception.Response
    if ($null -eq $response) {
        return $null
    }
    try {
        if ($null -ne $response.Content) {
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        } elseif ($response -is [System.Net.HttpWebResponse]) {
            $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
            try {
                $body = $reader.ReadToEnd()
            } finally {
                $reader.Dispose()
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($body)) {
            return $body | ConvertFrom-Json
        }
    } catch {
    }
    return $null
}

function Wait-OrchestratorReady {
    param(
        [string]$Url,
        [string]$Mode
    )
    $timeoutSeconds = if ($Mode -eq "local") { 240 } else { 20 }
    $deadline = [DateTime]::UtcNow.AddSeconds($timeoutSeconds)
    $attempt = 0
    $lastState = "service is not accepting connections"

    while ([DateTime]::UtcNow -lt $deadline) {
        $attempt += 1
        try {
            $payload = Invoke-RestMethod -Uri $Url -TimeoutSec 2
        } catch {
            $statusCode = Get-HttpStatusCode $_
            if ($null -eq $statusCode) {
                $lastState = "service is not accepting connections"
                Start-Sleep -Milliseconds 500
                continue
            }

            $errorPayload = Get-HttpErrorPayload $_
            $reason = if ($null -ne $errorPayload) { [string]$errorPayload.reason } else { "" }
            if ($statusCode -eq 503 -and $Mode -eq "local" -and $reason -eq "OLLAMA_WARMING") {
                $lastState = "OLLAMA_WARMING"
                if ($attempt -eq 1 -or $attempt % 10 -eq 0) {
                    Write-Host "Ollama model is warming; waiting up to 240 seconds..."
                }
                Start-Sleep -Milliseconds 500
                continue
            }

            $reasonDetail = if ([string]::IsNullOrWhiteSpace($reason)) { "no reason supplied" } else { $reason }
            throw "Orchestrator readiness failed: HTTP $statusCode ($reasonDetail)"
        }

        if ($payload.status -ne "ready") {
            throw "Orchestrator readiness returned unexpected status: $($payload.status)"
        }
        if ($Mode -eq "local") {
            if ($payload.provider_mode -ne "local" -or $payload.providers.ollama.ready -ne $true) {
                throw "Orchestrator readiness did not confirm the local Ollama provider"
            }
        }
        return
    }

    throw "Timed out after $timeoutSeconds seconds waiting for $Url ($lastState)"
}

function Wait-HttpSuccess {
    param([string]$Url)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return
            }
            throw "HTTP $($response.StatusCode)"
        } catch {
            $statusCode = Get-HttpStatusCode $_
            if ($null -ne $statusCode) {
                throw "HTTP endpoint failed: $Url returned $statusCode"
            }
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
    $env:VITE_MEDIA_MODE = switch ($ProviderMode) {
        "real" { "livekit" }
        "local" { "local" }
        default { "mock" }
    }
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
    Wait-OrchestratorReady -Url "http://127.0.0.1:8000/health/ready" -Mode $ProviderMode
    Wait-HttpSuccess "http://127.0.0.1:5173/"
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

Write-Host "PsyAvatar Care demo is ready in $ProviderMode mode."
Write-Host "User call:       http://127.0.0.1:5173/"
Write-Host "API docs:        http://127.0.0.1:8000/docs"
Write-Host "Clinician view:  http://127.0.0.1:5173/?view=clinician&session=<session_id>"
Write-Host "Process IDs are recorded in runtime/demo-processes.json."
