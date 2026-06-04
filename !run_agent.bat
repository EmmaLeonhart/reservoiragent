@echo off
REM run_agent.bat — local debug/demo launcher (NOT the distributable; that's the .exe).
REM Loads the most recent top reservoir-agent model, runs the recall demo, then a REPL.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

set "PY=python"
if exist "C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe" set "PY=C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe"
if defined RESERVOIR_PYTHON set "PY=%RESERVOIR_PYTHON%"

echo Using Python: %PY%
"%PY%" -m reservoir.installer %*
endlocal
