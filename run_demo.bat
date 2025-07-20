@echo off
REM One-liner RAG demo for Windows (x86 + OpenVINO)
REM Usage: run_demo.bat "your query"

IF "%~1"=="" (
  ECHO Error: Please provide a query.
  ECHO Usage: run_demo.bat "Your question about the Procyon Guide"
  EXIT /B 1
)

REM Create venv if missing
IF NOT EXIST ".venv\" (
  ECHO Creating virtual environment...
  python -m venv .venv
)

REM Activate venv
CALL .venv\Scripts\activate

REM Install dependencies
ECHO Installing dependencies...
python -m pip install --upgrade pip >nul
pip install -q --exists-action i -r requirements.txt

REM Run RAG CLI
python rag_cli.py --query %* --top_k 5