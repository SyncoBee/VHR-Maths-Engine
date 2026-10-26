@echo off
:: VHR Maths Engine Pro - 1-Click Launch & Windows Unblocker
:: Published and Created by Vihanga Himantha

title VHR Maths Engine Pro Launcher
cd /d "%~dp0"

echo ===================================================
echo   VHR Maths Engine Pro v4.0
echo   Published by Vihanga Himantha
echo ===================================================
echo.
echo [*] Checking Windows security permissions...

:: Remove Mark of the Web (Zone.Identifier) from all files in this directory
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path . -Recurse | Unblock-File -ErrorAction SilentlyContinue"

if exist "VHR_Maths_Engine.exe" (
    echo [+] Launching VHR Maths Engine Pro...
    start "" "VHR_Maths_Engine.exe"
) else if exist "main.py" (
    echo [+] Launching from Python source...
    python main.py
) else (
    echo [!] Could not locate VHR_Maths_Engine.exe or main.py.
    pause
)
