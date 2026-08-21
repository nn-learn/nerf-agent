[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$orchestratorRoot = Join-Path $projectRoot "services\orchestrator"
$avatarRoot = Join-Path $projectRoot "services\avatar_radnerf"
$webRoot = Join-Path $projectRoot "apps\web"
$python = Join-Path $orchestratorRoot ".venv\Scripts\python.exe"
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source

function Invoke-Verification {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [scriptblock]$Command
    )
    Write-Host "`n== $Name =="
    Push-Location $WorkingDirectory
    try {
        & $Command
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

Invoke-Verification "PowerShell launcher tests" $projectRoot {
    $result = Invoke-Pester `
        (Join-Path $projectRoot "scripts\tests\doctor.Tests.ps1") `
        -PassThru
    if ($result.FailedCount -gt 0) {
        throw "PowerShell launcher tests failed"
    }
    $global:LASTEXITCODE = 0
}
Invoke-Verification "Orchestrator tests" $orchestratorRoot {
    & $python -m pytest -q
}
Invoke-Verification "Orchestrator lint" $orchestratorRoot {
    & $python -m ruff check app tests
}
Invoke-Verification "Orchestrator types" $orchestratorRoot {
    & $python -m mypy app
}
Invoke-Verification "RAD-NeRF CPU contract tests" $avatarRoot {
    & $python -m pytest tests -q
}
Invoke-Verification "RAD-NeRF lint" $avatarRoot {
    & $python -m ruff check avatar tests
}
Invoke-Verification "RAD-NeRF types" $avatarRoot {
    & $python -m mypy avatar
}
Invoke-Verification "Web unit tests" $webRoot {
    & $npm run test -- --run
}
Invoke-Verification "Web production build" $webRoot {
    & $npm run build
}
Invoke-Verification "Fixed acceptance report" $orchestratorRoot {
    & $python -m app.evals.report `
        --risk (Join-Path $projectRoot "evals\risk_cases.jsonl") `
        --vision (Join-Path $projectRoot "evals\vision_cases.jsonl")
}
Invoke-Verification "Agent V3 release gate" $orchestratorRoot {
    & $python -m app.evals.run_agent_v3 `
        --cases (Join-Path $projectRoot "evals\agent_v3_cases.jsonl")
}
Invoke-Verification "Agent V4 release gate" $orchestratorRoot {
    & $python -m app.evals.run_agent_v4 `
        --scenarios (Join-Path $projectRoot "evals\agent_v4_scenarios.jsonl")
}
Invoke-Verification "Memory V5 release gate" $orchestratorRoot {
    & $python -m app.memory.run_v5_evaluation `
        --cases (Join-Path $projectRoot "evals\memory_v5_cases.jsonl")
}

Write-Host "`nOptional browser E2E and CUDA smoke are not run by the zero-GPU baseline."
Write-Host "Verification complete."
