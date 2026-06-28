[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$RunDir = ".run",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunPath = Join-Path $RepoRoot $RunDir
$Launcher = Join-Path $PSScriptRoot "start_background.py"

New-Item -ItemType Directory -Force $RunPath | Out-Null

& $Python $Launcher --run-dir $RunDir -- @AppArgs
exit $LASTEXITCODE
