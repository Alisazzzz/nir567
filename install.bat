@echo off

echo =========================================
echo Installing project NIR
echo =========================================
echo.

echo Checking Python 3.13...
py -3.13 --version > nul 2>&1

if errorlevel 1 (
    echo ERROR: Python 3.13 is not installed.
    echo Install Python 3.13 first:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
py -3.13 -m venv .venv

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Updating pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Installing spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo Installing spaCy Russian model...
python -m spacy download ru_core_news_sm

echo.
echo =========================================
echo Installation completed successfully!
echo =========================================
echo.

echo To run the project later:
echo.
echo cd nir567
echo .venv\Scripts\activate
echo python main.py
echo.

pause