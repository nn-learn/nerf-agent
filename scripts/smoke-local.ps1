[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$ApiBaseUrl = $ApiBaseUrl.TrimEnd("/")
$expectedModel = $env:PSYAVATAR_TEXT_MODEL
if ([string]::IsNullOrWhiteSpace($expectedModel)) {
    $expectedModel = "qwen3.6:latest"
}
$prompt = -join (0x6700, 0x8FD1, 0x5DE5, 0x4F5C, 0x538B, 0x529B, 0x6709, 0x70B9, 0x5927, 0xFF0C, 0x6211, 0x60F3, 0x5148, 0x68B3, 0x7406, 0x4E00, 0x4E0B, 0x3002 | ForEach-Object { [char]$_ })
$knownMockSentence = -join (0x6211, 0x542C, 0x89C1, 0x4F60, 0x6B63, 0x5728, 0x627F, 0x53D7, 0x538B, 0x529B, 0xFF0C, 0x6211, 0x4EEC, 0x53EF, 0x4EE5, 0x6162, 0x6162, 0x8BF4, 0x3002 | ForEach-Object { [char]$_ })
$sessionId = $null
$headers = $null
$operationFailure = $null
$cleanupFailure = $null
$readinessDurationMs = 0
$sessionDurationMs = 0
$turnDurationMs = 0
$cleanupDurationMs = 0
$turnStatus = $null
$overall = [System.Diagnostics.Stopwatch]::StartNew()

try {
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $readiness = Invoke-RestMethod -Method Get -Uri "$ApiBaseUrl/health/ready" -TimeoutSec 15
    } catch {
        throw "Local readiness request failed"
    } finally {
        $timer.Stop()
        $readinessDurationMs = $timer.ElapsedMilliseconds
    }
    if (
        $readiness.status -ne "ready" -or
        $readiness.provider_mode -ne "local" -or
        $readiness.providers.ollama.ready -ne $true -or
        $readiness.providers.ollama.model -ne $expectedModel
    ) {
        throw "Readiness did not confirm exact local Ollama model $expectedModel"
    }

    $timer.Restart()
    try {
        $session = Invoke-RestMethod `
            -Method Post `
            -Uri "$ApiBaseUrl/api/sessions" `
            -ContentType "application/json" `
            -Body (@{ camera_consent = $false } | ConvertTo-Json -Compress) `
            -TimeoutSec 15
    } catch {
        throw "Session creation failed"
    } finally {
        $timer.Stop()
        $sessionDurationMs = $timer.ElapsedMilliseconds
    }
    $sessionId = [string]$session.session_id
    $accessToken = [string]$session.access_token
    if (
        [string]::IsNullOrWhiteSpace($sessionId) -or
        [string]::IsNullOrWhiteSpace($accessToken)
    ) {
        throw "Session response did not contain the required credentials"
    }
    $headers = @{ Authorization = "Bearer $accessToken" }
    $turnBody = @{ text = $prompt; visual_summary = "" } | ConvertTo-Json -Compress
    $turnBodyBytes = [System.Text.Encoding]::UTF8.GetBytes($turnBody)

    $timer.Restart()
    try {
        $turn = Invoke-RestMethod `
            -Method Post `
            -Uri "$ApiBaseUrl/api/sessions/$([Uri]::EscapeDataString($sessionId))/turns" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body $turnBodyBytes `
            -TimeoutSec 240
    } catch {
        throw "Typed local turn failed"
    } finally {
        $timer.Stop()
        $turnDurationMs = $timer.ElapsedMilliseconds
    }

    $turnStatus = [string]$turn.status
    $displayText = [string]$turn.response.display_text
    if ($turnStatus -ne "completed") {
        throw "Typed local turn did not complete"
    }
    if ([string]::IsNullOrWhiteSpace($displayText)) {
        throw "Typed local turn returned no display text"
    }
    if ($displayText -eq $knownMockSentence) {
        throw "Typed local turn returned the known mock response"
    }
} catch {
    $operationFailure = $_.Exception.Message
} finally {
    if (
        -not [string]::IsNullOrWhiteSpace($sessionId) -and
        $null -ne $headers
    ) {
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $null = Invoke-RestMethod `
                -Method Delete `
                -Uri "$ApiBaseUrl/api/sessions/$([Uri]::EscapeDataString($sessionId))" `
                -Headers $headers `
                -TimeoutSec 15
        } catch {
            $cleanupFailure = "Session cleanup failed"
        } finally {
            $timer.Stop()
            $cleanupDurationMs = $timer.ElapsedMilliseconds
        }
    }
    $overall.Stop()
}

if ($null -ne $cleanupFailure) {
    throw $cleanupFailure
}
if ($null -ne $operationFailure) {
    throw $operationFailure
}

Write-Host "Local smoke status: $turnStatus"
Write-Host "Readiness:       $readinessDurationMs ms"
Write-Host "Session create:  $sessionDurationMs ms"
Write-Host "Typed turn:      $turnDurationMs ms"
Write-Host "Session cleanup: $cleanupDurationMs ms"
Write-Host "Total:           $($overall.ElapsedMilliseconds) ms"
