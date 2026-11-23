@echo off
REM ===== Starts central dashboard + edge service, opens both in the browser =====
cd /d "%~dp0"
if not exist models\demand_rf.joblib (echo Model not found - run setup.bat first. & pause & exit /b 1)
start "CENTRAL DASHBOARD :8050" cmd /k ".venv\Scripts\activate.bat && python dashboard\app.py"
timeout /t 3 /nobreak >nul
start "EDGE SERVICE :5000" cmd /k ".venv\Scripts\activate.bat && python edge_service\app.py"
timeout /t 6 /nobreak >nul
start "" http://localhost:5000
start "" http://localhost:8050
echo Edge UI: http://localhost:5000    Dashboard: http://localhost:8050
echo Close the two console windows to stop.
