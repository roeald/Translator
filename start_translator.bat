@echo off
setlocal
cd /d "%~dp0"
title Offline Formatted PDF Translator (pdf2zh + NLLB)

echo ===================================================
echo   Checking Python ^& Offline Translation Setup...
echo ===================================================

REM pdf2zh unterstuetzt nur Python 3.10 - 3.12 (NICHT 3.13), daher Python 3.12 verwenden
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python 3.12 not found. Installing automatically...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo [!] Please close this window and double-click start_translator.bat again to finish setup.
    pause
    exit /b
)

REM Eigene virtuelle Umgebung, damit die Pakete sich nicht mit anderen Installationen mischen
if not exist ".venv\Scripts\python.exe" (
    echo [+] Creating virtual environment...
    py -3.12 -m venv .venv
)
set "PY=%~dp0.venv\Scripts\python.exe"

echo [+] Checking required libraries (pdf2zh, Transformers, Torch, Flask)...
"%PY%" -m pip install -q --upgrade pip
echo [i] First run: several GB are downloaded, this can take 5-15 minutes. Progress is shown below.
REM Versions-Pins, weil pdf2zh 1.x mit den neuesten Paketen nicht mehr startet:
REM   numpy unter 2.3  : np.fromstring (Binaermodus) wurde in NumPy 2.3 entfernt
REM   tencentcloud-tmt  : neuere Versionen haben TextTranslate entfernt
REM   bitstring unter 5: Version 5 hat die Klasse BitStream entfernt (babeldoc braucht sie)
"%PY%" -m pip install "numpy<2.3" "bitstring<5" "tencentcloud-sdk-python-tmt==3.0.1269" pdf2zh transformers torch sentencepiece flask
if %errorlevel% neq 0 (
    echo [!] Installation of libraries failed. See messages above.
    pause
    exit /b 1
)

REM OCR fuer Text in Bildern: bewusst getrennt installiert, damit ein Problem hier die normale Uebersetzung nicht blockiert.
REM Die Versions-Pins werden wiederholt, damit easyocr numpy oder bitstring nicht auf eine zu neue Version zieht.
echo [+] Checking OCR library for text inside images (easyocr)...
"%PY%" -m pip install easyocr "numpy<2.3" "bitstring<5"
if %errorlevel% neq 0 echo [!] easyocr could not be installed - text inside images will not be translated.
"%PY%" -c "import easyocr" >nul 2>&1
if %errorlevel% neq 0 echo [!] easyocr cannot be loaded - text inside images will not be translated.

set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo [+] Starting local offline translator...
REM Browser erst nach ein paar Sekunden oeffnen, damit der Server schon laeuft
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:5000"
"%PY%" app.py
pause
