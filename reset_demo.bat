@echo off
REM Wipes local edge store + dashboard history (stop the services first)
cd /d "%~dp0"
if exist edge_service\edge_store.db del edge_service\edge_store.db
if exist dashboard\central.db del dashboard\central.db
echo Demo data reset. Edge store re-seeds from data\ on next start.
pause
