@echo off
REM Usage: run_simulator.bat tests | pos | scan
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set MODE=%1
if "%MODE%"=="" set MODE=tests
python simulator\store_simulator.py %MODE% %2 %3 %4 %5
pause
