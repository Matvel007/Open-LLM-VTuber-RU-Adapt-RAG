@echo off
cd /d "%~dp0"
chcp 65001 >nul

echo Open-LLM-VTuber - Server
echo For desktop mode run: start_desktop.bat
echo.

:: Build frontend if source exists (updates frontend from dist/web)
if exist "frontend\package.json" (
    if exist "frontend\dist\web\index.html" (
        echo Updating frontend from build...
        xcopy /E /Y "frontend\dist\web\*" "frontend\" >nul
        echo Frontend updated.
    ) else (
        echo Building frontend...
        cd frontend
        call npm run build:web
        cd ..
        if exist "frontend\dist\web\index.html" (
            xcopy /E /Y "frontend\dist\web\*" "frontend\" >nul
            echo Frontend built and updated.
        )
    )
    echo.
)

uv run run_server.py %*
pause
