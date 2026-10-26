# VHR Maths Engine Pro Launcher
Set-Location -Path $PSScriptRoot

$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $python main.py $args
