Param(
  [string]$Path = (Get-Location).Path,
  [int]$DebounceMs = 1500
)

Write-Host "Watching for changes in: $Path" -ForegroundColor Cyan

$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $Path
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents = $true
$fsw.Filter = '*.*'

# Exclusions
$exclude = @('\.git\\', '\\.venv\\', '\\__pycache__\\')

# Debounce timer
$timer = New-Object System.Timers.Timer
$timer.Interval = $DebounceMs
$timer.AutoReset = $false

$pending = $false

$action = {
  $path = $Event.SourceEventArgs.FullPath
  if ($exclude | Where-Object { $path -match $_ }) { return }
  $script:pending = $true
  $timer.Stop(); $timer.Start()
}

$onTick = {
  if (-not $script:pending) { return }
  $script:pending = $false
  try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
      $env:Path += ';C:\\Program Files\\Git\\cmd'
    }
    git add -A | Out-Null
    $msg = "auto: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
      git commit -m $msg | Out-Null
      git push | Out-Null
      Write-Host "Committed and pushed: $msg" -ForegroundColor Green
    }
  } catch {
    Write-Warning $_
  }
}

Register-ObjectEvent $fsw Changed -Action $action | Out-Null
Register-ObjectEvent $fsw Created -Action $action | Out-Null
Register-ObjectEvent $fsw Deleted -Action $action | Out-Null
Register-ObjectEvent $fsw Renamed -Action $action | Out-Null
Register-ObjectEvent $timer Elapsed -Action $onTick | Out-Null

Write-Host "Auto-push active. Press Ctrl+C to stop." -ForegroundColor Yellow
while ($true) { Start-Sleep -Seconds 1 }

