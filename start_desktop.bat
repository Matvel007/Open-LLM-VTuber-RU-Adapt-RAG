@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
title Open-LLM-VTuber

echo ========================================
echo   Open-LLM-VTuber - Desktop mode
echo ========================================
echo.

:: Start server in separate window
echo [1/2] Starting server...
set "ROOT=%~dp0"
start "OLV-Server" /min cmd /c "cd /d %ROOT% && uv run run_server.py"
if errorlevel 1 (
    echo Error: Failed to start server. Run: uv sync
    pause
    exit /b 1
)

:: Wait for server to be ready (poll until port responds)
echo [2/2] Waiting for server to start...
set "MAX_WAIT=60"
set "WAITED=0"
:wait_loop
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:12393' -UseBasicParsing -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto server_ready
timeout /t 2 /nobreak >nul
set /a WAITED+=2
if %WAITED% geq %MAX_WAIT% (
    echo Server did not start within %MAX_WAIT% sec. Launching client anyway...
    goto server_ready
)
echo   ... waiting (!WAITED!/%MAX_WAIT% sec)
goto wait_loop

:server_ready
echo Server is ready.

:: Find and launch desktop client (prefer win-unpacked: has ffmpeg.dll etc.)
set "DESKTOP_EXE="
set "DESKTOP_DIR="
if exist "frontend\release\1.2.1\win-unpacked\open-llm-vtuber-electron.exe" (
    set "DESKTOP_EXE=frontend\release\1.2.1\win-unpacked\open-llm-vtuber-electron.exe"
    set "DESKTOP_DIR=frontend\release\1.2.1\win-unpacked"
)
if not defined DESKTOP_EXE if exist "desktop-app\open-llm-vtuber-electron.exe" if exist "desktop-app\ffmpeg.dll" set "DESKTOP_EXE=desktop-app\open-llm-vtuber-electron.exe"
if not defined DESKTOP_EXE if exist "desktop-app\Open LLM VTuber.exe" set "DESKTOP_EXE=desktop-app\Open LLM VTuber.exe"
if not defined DESKTOP_EXE (
    for %%F in ("desktop-app\*.exe") do if exist "desktop-app\ffmpeg.dll" set "DESKTOP_EXE=%%~fF"
    if defined DESKTOP_EXE goto launch
)

:launch
if defined DESKTOP_EXE (
    echo Launching desktop client...
    start "" "%DESKTOP_EXE%"
    echo.
    echo Done. Server in background, desktop client started.
) else (
    echo.
    echo Desktop client not found.
    echo Open browser: http://localhost:12393
    echo.
    echo To use desktop mode: cd frontend ^&^& npm run build:win
    echo Then run start_desktop.bat again.
    start "" "http://localhost:12393"
)

echo.
pause
