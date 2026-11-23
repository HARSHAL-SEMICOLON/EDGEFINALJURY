@echo off
REM ===== Smart Retail Edge - one-time setup (Windows) =====
cd /d "%~dp0"
echo [1/4] Creating virtual environment .venv ...
python -m venv .venv || (echo Python not found. Install Python 3.10+ and tick "Add to PATH". & pause & exit /b 1)
call .venv\Scripts\activate.bat
echo [2/4] Installing packages ...
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q || (echo Package install failed - check internet. & pause & exit /b 1)
echo [3/4] Generating dataset ...
python data\generate_dataset.py
echo [4/4] Training models (about 30 s) ...
python ml\train_model.py
if exist edge_service\edge_store.db del edge_service\edge_store.db
if exist dashboard\central.db del dashboard\central.db
echo.
echo Setup complete. Now double-click start_demo.bat
pause
