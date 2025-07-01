@echo off
echo Setting up TinyLlama Finance QLoRA Model...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Setup complete!
echo.
echo To test the model, run:
echo   python test_model.py
echo.
echo To retrain the model, run:
echo   python finetune_tinyllama_qlora_Version2.py
echo.
pause
