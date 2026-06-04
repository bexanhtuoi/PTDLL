$env:PYTHONPATH = "src"
Set-Location "C:\Users\PC\Downloads\PTDLL"
$logFile = "C:\Users\PC\Downloads\PTDLL\train_output.log"
$errFile = "C:\Users\PC\Downloads\PTDLL\train_err.log"
"Started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile
uv run python -m models.train *>&1 | Add-Content $logFile
"Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $logFile
