#!/bin/bash
# One-liner RAG demo for Linux/macOS (x86 + OpenVINO)
# Usage: ./run_demo.sh "your query"

if [ -z "$1" ]; then
  echo "Error: Please provide a query."
  echo "Usage: ./run_demo.sh \"Your question about the Procyon Guide\""
  exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install  -q --exists-action i -r requirements.txt

# Run RAG CLI
python rag_cli.py --query "$1" --top_k 5