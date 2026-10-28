@echo off
cd /d "%~dp0\.."
if not exist .env copy .env.example .env
cd backend
python -m pip install -q -r requirements.txt
python main.py
pause