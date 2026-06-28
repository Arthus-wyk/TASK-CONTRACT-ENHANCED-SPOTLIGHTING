[CmdletBinding()]
param(
    [string]$RunDir = ".run"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunPath = Join-Path $RepoRoot $RunDir
$PidFile = Join-Path $RunPath "main.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "No PID file found at $PidFile."
    exit 0
}

$pidText = (Get-Content $PidFile -Raw).Trim()
if (-not $pidText) {
    Set-Content -Path $PidFile -Value ""
    Write-Host "PID file is already empty."
    exit 0
}

$process = Get-Process -Id $pidText -ErrorAction SilentlyContinue
if (-not $process) {
    Set-Content -Path $PidFile -Value ""
    Write-Host "Process $pidText is not running. Cleared stale PID file."
    exit 0
}

Stop-Process -Id $pidText
Set-Content -Path $PidFile -Value ""
Write-Host "Stopped background run with PID $pidText."
