[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$RunDir = ".run",
    [string]$Model,
    [ValidateSet("no_defense", "spotlighting", "task_shield", "task-shield", "spotlighting_task_shield", "spotlighting-task-shield")]
    [string]$Defense,
    [string[]]$Suites,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunPath = Join-Path $RepoRoot $RunDir
$Launcher = Join-Path $PSScriptRoot "start_background.py"

New-Item -ItemType Directory -Force $RunPath | Out-Null

$LauncherArgs = @("--run-dir", $RunDir)
if ($Model) {
    $LauncherArgs += @("--model", $Model)
}
if ($Defense) {
    $LauncherArgs += @("--defense", $Defense)
}
if ($Suites -and $Suites.Count -gt 0) {
    $LauncherArgs += @("--suites")
    $LauncherArgs += $Suites
}

& $Python $Launcher @LauncherArgs -- @AppArgs
exit $LASTEXITCODE
