@echo off
REM Morning autopilot run. Invoked by Windows Task Scheduler at 09:00 daily.
REM Logs go to outputs\logs\autopilot-YYYYMMDD.log relative to repo root.

setlocal

REM Resolve repo root = parent of this script's directory
set "REPO=%~dp0.."
pushd "%REPO%"

REM Date stamp YYYYMMDD using PowerShell (avoids locale issues with %DATE%)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set STAMP=%%i

if not exist "outputs\logs" mkdir "outputs\logs"

set "LOG=outputs\logs\autopilot-%STAMP%.log"

echo [%date% %time%] starting autopilot >> "%LOG%"

REM Prefer .venv if present, else system python
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" -m crypto_llm_alpaca.cli autopilot --max-trades 3 --max-positions 5 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo [%date% %time%] exit=%RC% >> "%LOG%"
popd
exit /b %RC%
