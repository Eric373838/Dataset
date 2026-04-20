@echo off
setlocal

cd /d %~dp0

if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-llm-trainer.txt

echo [INFO] Launching Professional LLM Trainer UI...
python app.py

endlocal
