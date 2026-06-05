@echo off
REM run_agent.bat — launches the real-time side-by-side Reservoir Agent (Electron app).
REM The earlier colour-recall demo + REPL now lives in run_recall_demo.bat.
setlocal
set PYTHONUTF8=1
if not defined RESERVOIR_PYTHON if exist "C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe" set "RESERVOIR_PYTHON=C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe"
cd /d "%~dp0app\electron"
if not exist node_modules (
  echo Installing Electron dependencies ^(first run downloads Electron^)...
  call npm install || goto :err
)
echo Starting Reservoir Agent ...
call npm start
endlocal
goto :eof
:err
echo.
echo npm install failed. Ensure Node.js is installed and on PATH.
endlocal
exit /b 1
