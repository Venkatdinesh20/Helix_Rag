@echo off
echo ==========================================
echo  RAG Pipeline - First-time Setup
echo ==========================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12 from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv .venv

echo [2/4] Installing dependencies...
.venv\Scripts\pip install --upgrade pip --quiet
.venv\Scripts\pip install -r requirements.txt --quiet
echo       Done.

echo [3/4] Setting up .env file...
if not exist .env (
    echo OPENAI_API_KEY=your-api-key-here > .env
    echo OPENAI_MODEL=gpt-4o-mini >> .env
    echo ENVIRONMENT=local >> .env
    echo.
    echo  ** Open .env and replace 'your-api-key-here' with your OpenAI API key **
    echo  ** Get a key at: https://platform.openai.com/api-keys                 **
    echo.
) else (
    echo       .env already exists - skipping.
)

echo [4/4] Setup complete!
echo.
echo To start the application, run:
echo   run.bat
echo.
pause
