@echo off
setlocal

cd /d %~dp0

if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip

echo [INFO] Installing CUDA-enabled PyTorch build (cu121)...
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [INFO] Installing trainer dependencies...
python -m pip install -r requirements-llm-trainer.txt

echo [INFO] Launching Own LLM Pretraining Studio...
python app.py

endlocal
