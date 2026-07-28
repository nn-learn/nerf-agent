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
    & $venvPython -m pip install -e ".[dev]"
    & $venvPython -m pytest tests -q
}
finally {
    Pop-Location
}
