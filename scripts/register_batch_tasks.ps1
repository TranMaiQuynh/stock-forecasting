param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$StateDatabase,
    [ValidatePattern('^[A-Za-z0-9_-]{1,60}$')][string]$TaskPrefix = 'StockForecasting-Pilot',
    [string]$WorkingDirectory = (Split-Path -Parent $PSScriptRoot),
    [switch]$Register
)

$ErrorActionPreference = 'Stop'
$batchPython = (Resolve-Path -LiteralPath $Python).Path
$batchState = (Resolve-Path -LiteralPath $StateDatabase).Path
$batchWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$batchWrapper = Join-Path $PSScriptRoot 'run_batch_task.ps1'
$batchShell = (Get-Process -Id $PID).Path
foreach ($batchPath in @($batchPython, $batchState, $batchWorkingDirectory, $batchWrapper)) {
    if ($batchPath.Contains('"') -or $batchPath.Contains("`n")) { throw 'Invalid path for scheduled task' }
}
$batchTasks = @(
    @{ Name = ($TaskPrefix + '-Run'); Operation = 'run'; Minutes = 15 },
    @{ Name = ($TaskPrefix + '-Health'); Operation = 'health'; Minutes = 5 }
)
$batchPlan = @{
    python = $batchPython; state_database = $batchState; working_directory = $batchWorkingDirectory
    user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    logon_type = 'Interactive'; run_level = 'Limited'; tasks = $batchTasks; register_requested = [bool]$Register
}
if (-not $Register) { $batchPlan | ConvertTo-Json -Depth 5; exit 0 }

# Never overwrite an existing task, including tasks from another installation.
foreach ($batchTask in $batchTasks) {
    if (Get-ScheduledTask -TaskName $batchTask.Name -ErrorAction SilentlyContinue) {
        throw ('Task already exists: ' + $batchTask.Name)
    }
}
$batchPrincipal = New-ScheduledTaskPrincipal -UserId $batchPlan.user -LogonType Interactive -RunLevel Limited
$batchSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$batchCreated = @()
try {
    foreach ($batchTask in $batchTasks) {
        $batchArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "{0}" -Python "{1}" -StateDatabase "{2}" -Operation {3} -WorkingDirectory "{4}"' -f $batchWrapper, $batchPython, $batchState, $batchTask.Operation, $batchWorkingDirectory
        $batchAction = New-ScheduledTaskAction -Execute $batchShell -Argument $batchArguments -WorkingDirectory $batchWorkingDirectory
        $batchTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $batchTask.Minutes)
        Register-ScheduledTask -TaskName $batchTask.Name -Action $batchAction -Trigger $batchTrigger -Settings $batchSettings -Principal $batchPrincipal -Description 'Local research pilot; no training, model promotion, external messaging or trading.' | Out-Null
        $batchCreated += $batchTask.Name
    }
} catch {
    foreach ($batchCreatedName in $batchCreated) {
        Unregister-ScheduledTask -TaskName $batchCreatedName -Confirm:$false
    }
    throw
}
$batchPlan | ConvertTo-Json -Depth 5
