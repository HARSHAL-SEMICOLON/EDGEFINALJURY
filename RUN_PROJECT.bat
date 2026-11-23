@echo off
title Smart Retail Edge - Launcher
REM =====================================================================
REM  Smart Retail Product Demand Prediction System - ONE-CLICK LAUNCHER
REM  First run : creates .venv, installs packages, builds data, trains model
REM  Next runs : skips setup and starts straight away (works offline)
REM =====================================================================
cd /d "%~dp0"

echo.
echo  ==============================================================
echo    SMART RETAIL PRODUCT DEMAND PREDICTION  -  EDGE ANALYTICS
echo  ==============================================================
echo.

REM ---------- 1. Python check ----------
where python >nul 2>&1
if errorlevel 1 (
    echo  [X] Python not found.
    echo      Install Python 3.10+ from python.org and tick "Add Python to PATH".
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v found

REM ---------- 2. Virtual environment ----------
if not exist ".venv\Scripts\activate.bat" (
    echo  [..] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 ( echo  [X] Could not create .venv & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"
echo  [OK] Virtual environment active

REM ---------- 3. Packages ----------
python -c "import flask, sklearn, pandas, numpy, joblib, matplotlib, requests, paho.mqtt, pytest" >nul 2>&1
if errorlevel 1 (
    echo  [..] Installing packages - needs internet, 2-3 min ...
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt -q
    if errorlevel 1 ( echo  [X] Package install failed - check internet. & pause & exit /b 1 )
)
echo  [OK] Packages installed

REM ---------- 4. Dataset ----------
if not exist "data\retail_sales.csv" (
    echo  [..] Generating dataset ...
    python data\generate_dataset.py
    if errorlevel 1 ( echo  [X] Dataset generation failed & pause & exit /b 1 )
)
echo  [OK] Dataset ready

REM ---------- 5. Model ----------
if not exist "models\demand_rf.joblib" (
    echo  [..] Training models - about 30-60 s ...
    python ml\train_model.py
    if errorlevel 1 ( echo  [X] Training failed & pause & exit /b 1 )
    if exist edge_service\edge_store.db del edge_service\edge_store.db
    if exist dashboard\central.db del dashboard\central.db
)
echo  [OK] Trained Random Forest model ready

:START
REM ---------- 6. Start services ----------
echo.
echo  [..] Starting Central Dashboard  (port 8050) ...
start "CENTRAL DASHBOARD :8050" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && python dashboard\app.py"
timeout /t 3 /nobreak >nul
echo  [..] Starting Edge Service       (port 5000) ...
start "EDGE SERVICE :5000" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && python edge_service\app.py"

REM wait until the edge service answers (max ~40 s)
set /a tries=0
:WAIT
timeout /t 2 /nobreak >nul
set /a tries+=1
curl -s -o nul http://localhost:5000/api/health >nul 2>&1
if not errorlevel 1 goto READY
if %tries% lss 20 goto WAIT
echo  [!] Edge service is slow to start - check the EDGE SERVICE window.
:READY
start "" http://localhost:5000
start "" http://localhost:8050
echo  [OK] Edge UI   : http://localhost:5000
echo  [OK] Dashboard : http://localhost:8050

REM ---------- 7. Demo menu ----------
:MENU
echo.
echo  ------------------------- DEMO MENU -------------------------
echo   1. Run 19 test cases  (normal / abnormal / invalid inputs)
echo   2. Stream live POS sales  (STR01, 3 days)
echo   3. Batch CSV prediction  (file with errors)
echo   4. Run 26 automated tests  (pytest)
echo   5. Open Edge UI + Dashboard in browser again
echo   6. Reset demo data  (stops services first)
echo   7. Restart services
echo   0. Exit  (services keep running - close their windows to stop)
echo  -------------------------------------------------------------
set "choice="
set /p choice=  Choose an option: 

if "%choice%"=="1" ( python simulator\store_simulator.py tests & goto MENU )
if "%choice%"=="2" ( python simulator\store_simulator.py pos --store STR01 --days 3 & goto MENU )
if "%choice%"=="3" ( python simulator\store_simulator.py batch --file data\sample_batch_with_errors.csv & goto MENU )
if "%choice%"=="4" ( python -m pytest -v tests & goto MENU )
if "%choice%"=="5" ( start "" http://localhost:5000 & start "" http://localhost:8050 & goto MENU )
if "%choice%"=="6" goto RESET
if "%choice%"=="7" goto RESTART
if "%choice%"=="0" exit /b 0
echo  Invalid option.
goto MENU

:RESET
call :STOP
if exist edge_service\edge_store.db del edge_service\edge_store.db
if exist dashboard\central.db del dashboard\central.db
echo  [OK] Demo data reset. Choose 7 to start the services again.
goto MENU

:RESTART
call :STOP
goto START

:STOP
echo  [..] Stopping services ...
taskkill /fi "WINDOWTITLE eq CENTRAL DASHBOARD :8050*" /t /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq EDGE SERVICE :5000*" /t /f >nul 2>&1
timeout /t 2 /nobreak >nul
exit /b 0
