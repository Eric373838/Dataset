@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d %~dp0

echo ======================================================
echo   Own LLM Pretraining Studio - Auto Setup + Launch
echo ======================================================

set "PYTHON_EXE="
where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_EXE=py -3"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python 3.10+ was not found.
    echo Install Python and enable "Add Python to PATH", then run again.
    goto :fail
)

echo [INFO] Using Python launcher: %PYTHON_EXE%

if not exist .venv (
    echo [INFO] Creating virtual environment (.venv)...
    call %PYTHON_EXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :fail
    )
)

call .venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv
    goto :fail
)

echo [INFO] Upgrading packaging tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel.
    goto :fail
)

echo [INFO] Installing CUDA-enabled PyTorch (tries cu124, then cu121)...
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARN] cu124 install failed, trying cu121...
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERROR] Failed to install CUDA-enabled PyTorch wheels.
        echo Check internet/proxy/SSL and NVIDIA driver compatibility.
        goto :fail
    )
)

echo [INFO] Installing trainer dependencies...
python -m pip install --upgrade -r requirements-llm-trainer.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements-llm-trainer.txt
    goto :fail
)

echo [INFO] Verifying key imports...
python -c "import torch,transformers,gradio,datasets,tokenizers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] Environment verification failed.
    goto :fail
)

echo [INFO] Launching app.py ...
python app.py
if errorlevel 1 (
    echo [ERROR] app.py exited with an error.
    goto :fail
)

echo [INFO] App closed normally.
goto :end

:fail
echo.
echo [FAILED] Setup or launch failed. See messages above.
echo Press any key to close...
pause >nul
exit /b 1

:end
echo.
echo Press any key to close...
pause >nul
endlocal
