@echo off
title Push Smart Retail Edge to GitHub
REM =====================================================================
REM  Pushes this project to https://github.com/HARSHAL-SEMICOLON/EDGEFINALJURY
REM  Uses YOUR GitHub login (a sign-in window opens the first time).
REM  .gitignore keeps out .venv, *.db and the trained model.
REM =====================================================================
cd /d "%~dp0"
set "REPO=https://github.com/HARSHAL-SEMICOLON/EDGEFINALJURY.git"

where git >nul 2>&1
if errorlevel 1 (
    echo  [X] Git is not installed. Get it from https://git-scm.com/download/win
    echo      then run this file again.
    pause & exit /b 1
)

REM first-time identity (only if not set)
git config --global user.name >nul 2>&1 || git config --global user.name "HARSHAL-SEMICOLON"
git config --global user.email >nul 2>&1 || git config --global user.email "HARSHAL-SEMICOLON@users.noreply.github.com"

if not exist ".git" (
    echo  [..] Initialising git repository ...
    git init -q
)
git branch -M main

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin %REPO%
) else (
    git remote set-url origin %REPO%
)
echo  [OK] Remote: %REPO%

echo  [..] Committing files ...
git add -A
git diff --cached --quiet
if errorlevel 1 (
    git commit -q -m "Smart Retail Product Demand Prediction - Edge Analytics jury project"
    echo  [OK] Commit created
) else (
    echo  [OK] Nothing new to commit
)

echo  [..] Syncing with GitHub (sign in if a window opens) ...
git pull origin main --allow-unrelated-histories --no-edit -q >nul 2>&1

git push -u origin main
if errorlevel 1 (
    echo.
    echo  [X] Push failed. Check: you are signed in to the HARSHAL-SEMICOLON account,
    echo      and the repo EDGEFINALJURY exists on GitHub.
    pause & exit /b 1
)
echo.
echo  [OK] Done!  Open: https://github.com/HARSHAL-SEMICOLON/EDGEFINALJURY
start "" https://github.com/HARSHAL-SEMICOLON/EDGEFINALJURY
pause
