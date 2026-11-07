@echo off
rem Double-click to start the normal app (original chord model) on port 8764, to compare with "Start Version A.bat".
rem (Port 8764, because the Chordify clone uses 8765.) Keep this window open while you use it; close it to stop.
title Nathan Chordify - Original (port 8764)
cd /d "%~dp0"
echo Starting the original app... your browser will open in a few seconds.
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start "" http://127.0.0.1:8764"
".venv\Scripts\python.exe" -m nathan_chordify.server --port 8764
pause
