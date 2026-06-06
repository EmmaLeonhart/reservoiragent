@echo off
REM !runAgent.bat — top-of-folder one-click launcher for the real-time Reservoir Agent.
REM The "!" prefix sorts it to the top of the folder so it's the obvious thing to run.
REM This OPENS THE ELECTRON APP. The implementation lives in run_agent.bat (single source
REM of truth); this wrapper just forwards to it so there is nothing to keep in sync.
REM (The old !run_agent.bat ran the installer console REPL — that lives in run_recall_demo.bat.)
call "%~dp0run_agent.bat" %*
