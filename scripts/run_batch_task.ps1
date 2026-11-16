param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$StateDatabase,
    [ValidateSet('run', 'health')][string]$Operation = 'run',
    [string]$WorkingDirectory = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$batchPython = (Resolve-Path -LiteralPath $Python).Path
$batchState = (Resolve-Path -LiteralPath $StateDatabase).Path
$batchWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$env:GIT_CEILING_DIRECTORIES = Split-Path -Parent (Split-Path -Parent $batchPython)
if ($batchState.Contains('"') -or $batchState.Contains("`n")) { throw 'Invalid state database path' }
$batchLogDirectory = Join-Path (Split-Path -Parent $batchState) 'task_logs'
New-Item -ItemType Directory -Force -Path $batchLogDirectory | Out-Null
$batchLogId = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$batchArguments = '-m quant_forecast_lab.cli batch {0} --state-db "{1}" --mode live' -f $Operation, $batchState
$batchProcess = Start-Process -FilePath $batchPython -ArgumentList $batchArguments -WorkingDirectory $batchWorkingDirectory -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $batchLogDirectory ($batchLogId + '.out.txt')) -RedirectStandardError (Join-Path $batchLogDirectory ($batchLogId + '.err.txt'))
exit $batchProcess.ExitCode
