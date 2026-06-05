@echo off
REM run_recall_demo.bat — the cross-pass colour-recall demo + stateful REPL (GPT-2 artifact).
REM This is the trained feasibility model from the paper's substrate result. The live
REM real-time agent app is run_agent.bat instead.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe" set "PY=C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe"
if defined RESERVOIR_PYTHON set "PY=%RESERVOIR_PYTHON%"
"%PY%" -m reservoir.installer %*
endlocal
