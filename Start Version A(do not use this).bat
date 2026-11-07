@echo off
rem Double-click to try "version A" (the fine-tuned ACE model ft1-a) in its own copy of the app.
rem The normal app and its songs are not changed. Keep this window open while you use it; close it to stop.
title Nathan Chordify - Version A (port 8766)
cd /d "%~dp0"
echo Starting Version A... the first start analyses your songs again (a few minutes), then your browser opens.
".venv\Scripts\python.exe" scripts\try_ace_weights.py --run ft1-a --port 8766
pause
