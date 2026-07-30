param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$serviceRoot = Join-Path $projectRoot "services\orchestrator"
$venvPython = Join-Path $serviceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$version = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($version -ne "3.12") {
    throw "PsyAvatar Orchestrator requires Python 3.12, got $version"
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $PythonExe -m venv (Join-Path $serviceRoot ".venv")
}

Push-Location $serviceRoot
try {
    & $venvPython -m pip install -e ".[dev,speech,rag]"
    if ($LASTEXITCODE -ne 0) {
        throw "Orchestrator dependency installation failed"
    }
    & $venvPython -c "import edge_tts, faster_whisper, sentence_transformers, webrtcvad"
    if ($LASTEXITCODE -ne 0) {
        throw "Speech/RAG dependency import check failed"
    }
    & $venvPython -m pytest tests -q
    if ($LASTEXITCODE -ne 0) {
        throw "Orchestrator tests failed"
    }
}
finally {
    Pop-Location
}
