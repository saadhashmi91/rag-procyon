# 🧠 RAG over Procyon Guide using LLaMA 3.1 + OpenVINO

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that answers natural language questions about the **UL Procyon Benchmark User Guide** using a locally quantized version of **Meta LLaMA 3.1 8B Instruct**.

The system uses:
- `sentence-transformers` for semantic chunk embedding
- `FAISS` for similarity search over PDF text
- `OpenVINO` for efficient **INT4 quantized inference** of LLaMA 3.1
- A minimal CLI (`rag_cli.py`) to query the guide locally

---

## 📘 Background

UL Procyon is a suite of benchmark tools for measuring real-world PC performance across office productivity, AI, and media tasks. This tool allows you to semantically query the Procyon guide using natural language and receive accurate answers with citations from the guide text.

---

## 💻 Platform & Toolkit

| Platform       | Toolkit     | Purpose                                |
|----------------|-------------|----------------------------------------|
| x86-64 (Linux/Windows) | OpenVINO    | Quantized LLM inference (INT4 on CPU/GPU) |

---

## ⚙️ Installation & Setup

### 0. Pre-requisites:
- Python >=3.9 (Windows/Linux)
- Linux: python3.8-venv

```bash
sudo apt install python3.8-venv
```

### 1. Clone the repository

```bash
git clone https://github.com/saadhashmi91/rag_procyon.git
cd rag-procyon
```

### 2. Place the Procyon Guide

Add the PDF file here:
```
data/procyon_guide.pdf
```

### 3. Create a virtual environment and install dependencies

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bat
python -m venv .venv
call .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Step-by-Step Workflow

### 🔹 Step 1 – PDF Ingestion & Chunking

Extract and chunk the Procyon guide:

```bash
python ingest_pdf.py
```

### 🔹 Step 2 – Generate Embeddings + Build FAISS Index

```bash
python ingest_pdf.py  # Includes chunking + FAISS in one script
```

### 🔹 Step 3 – Download & Quantize LLaMA 3.1 (INT4)


Windows:
```bat
set HF_TOKEN=<Replace with your Huggingface token>
python convert_model_openvino.py
```

Linux/macOS:
```bash
export HF_TOKEN=<Replace with your Huggingface token>
python convert_model_openvino.py
```

> Downloads Meta LLaMA 3.1 8B Instruct and exports to OpenVINO IR INT4 format.

---

## 🚀 Querying the System

### ✅ One-liner Demo (Linux/macOS)

```bash
./run_demo.sh "What is the Office Productivity score and how is it calculated?"
```

### ✅ One-liner Demo (Windows)

```bat
run_demo.bat "What is the Office Productivity score and how is it calculated?"
```

### ✅ Direct CLI Usage

```bash
python rag_cli.py --query "How is benchmark score interpreted?" --top_k 5
```

### ✅ Chat Mode (Memory-enabled, In-session)

```bash
python rag_cli.py --chat
```

---

## 🧾 Command-Line Options

### `rag_cli.py`

| Option         | Description                                    |
|----------------|------------------------------------------------|
| `--query`      | Natural language query                         |
| `--top_k`      | Number of top chunks to retrieve (default: 5)  |
| `--chat`       | Interactive mode with memory (type `exit` to quit) |

---

## 🧠 Requirements

### ✅ Hardware
- x86-64 machine (Linux or Windows)
- ≥ 16 GB RAM recommended
- Intel CPU or GPU (OpenVINO backend)
- No GPU required — runs on CPU with quantized inference

### ✅ Dependencies

Install via:
```bash
pip install -r requirements.txt
```

Contents:
```
transformers==4.40.1
optimum==1.19.1
optimum-intel==1.14.0
openvino==2024.0.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
PyPDF2==3.0.1
nltk==3.8.1
```

---

## 📂 Project Structure

```
├── data/
│   ├── procyon_guide.pdf
│   ├── chunks.json
│   └── faiss_index/
├── models/
│   ├── llama3_fp16/
│   └── llama3_int4/
├── ingest_pdf.py
├── convert_model_openvino.py
├── rag_cli.py
├── run_demo.sh
├── run_demo.bat
├── requirements.txt
└── README.md
```

---

## ✅ License & Credits

- Procyon Guide © UL Solutions  
- LLaMA 3.1 © Meta Platforms, used under research license  
- OpenVINO™, Transformers, SentenceTransformers libraries

---

## 🙋 FAQ

> ❓ What if I don’t have a GPU?

✅ No problem — OpenVINO runs the LLaMA 3.1 model efficiently on CPU using INT4 quantization.

> ❓ Can I modify the guide or add more documents?

Yes! You can modify `ingest_pdf.py` to ingest additional documents or chain them with Procyon chunks.

---
