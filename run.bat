@echo off
rem Start DeFeat on Windows: http://localhost:8000
cd /d "%~dp0"
".venv\Scripts\python.exe" launch.py %*
