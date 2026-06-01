# GGUF LLM/VLM Notebook Backend for Colab and Kaggle

A clean, production-ready launcher and process manager designed to turn Google Colab T4 x1 and Kaggle T4 x2 GPU runtimes into lightweight, high-performance local GGUF inference backends.

It runs prebuilt `llama.cpp` CUDA binaries (zero source compile), supports multimodal (image input) and text GGUF models, exposes an OpenAI-compatible HTTP interface, and establishes public tunnels with automatic fallback mechanisms.

---

## 🚀 Key Features

* **Zero-Compile prebuilt CUDA Binary:** Instantly pulls compatible driver-linked `llama-server` CUDA 12 binaries, skipping the 30-minute compiling process.
* **Live Segmented Downloader:** Orchestrates multiple segmented downloads via `aria2c`, processing output into a clean, single-panel progress card inside Jupyter without scroll spam.
* **Dual-GPU Orchestration (Kaggle T4 x2):** Automatically coordinates horizontal (`row`) split modes for dual T4 cards, falling back to `layer` split or single-GPU modes seamlessly.
* **Multi-Tunnel Auto-Router:** Publicly exposes local endpoints over Ngrok or Cloudflare Quick Tunnel (no auth token required for Cloudflare fallback).
* **Decoupled Architecture:** Delegates heavy execution logic to the modular `gguf_backend` package, keeping Jupyter cells short, pure, and clean.

---

## 📂 Repository Layout

```text
llama-gguf-notebook-backend/
  README.md
  pyproject.toml
  gguf_backend/
    __init__.py
    config.py
    ui.py
    shell.py
    installer.py
    downloader.py
    server.py
    tunnel.py
    client.py
    diagnostics.py
  notebooks/
    kaggle_t4x2_backend.ipynb
    colab_t4x1_backend.ipynb
  docs/
    troubleshooting.md
    model_compatibility.md
```

---

## 🛠️ Step-by-Step Notebook Workflow (7 Cells)

Each notebook is designed around exactly **7 core cells**:

1. **Cell 1 — Import Backend:** Clones the repository and imports the `gguf_backend` package modules.
2. **Cell 2 — Run Diagnostics:** Runs checks on available GPUs, VRAM, CUDA versions, storage space, and internet connections.
3. **Cell 3 — Install Binaries:** Downloads and extracts precompiled `llama.cpp` CUDA binaries and configures local tunnel executables.
4. **Cell 4 — Parallel Download:** Downloads the specified GGUF model and optional VLM projector via segmented connections.
5. **Cell 5 — Start Server:** Starts `llama-server` in the background with profile-specific GPU splitting, waits for the health check to succeed, and runs a warmup query.
6. **Cell 6 — Local Verification:** Sends a completion request to `http://127.0.0.1:8080/v1/chat/completions` and prints latency metrics.
7. **Cell 7 — Public Tunnel:** Launches Ngrok or Cloudflare Quick Tunnel, outputs the public endpoints, and runs public end-to-end completions verification.

---

## ⚡ Default Server Profiles

### 1. Google Colab (Tesla T4 x1)
* **Default Context Size:** 4096 tokens
* **Split Mode:** `none` (Single GPU 0)
* **Flash Attention:** Enabled
* **Cache KV Types:** `f16` (User override to `q8_0` recommended to save VRAM on larger models)

### 2. Kaggle (Tesla T4 x2)
* **Default Context Size:** 8192 tokens
* **Split Mode:** `row` (Horizontal tensor split, automatically falls back to `layer` split on failure)
* **Tensor Split Ratio:** `"1,1"` (Equally distributed VRAM)
* **Flash Attention:** Enabled
* **Cache KV Types:** `f16`

---

## 📚 Advanced Documentation

Detailed information can be found in the following docs:
* Check out [Troubleshooting Guide](docs/troubleshooting.md) for OOMs, CUDA mismatches, and Tunnel connection issues.
* Check out [Model Compatibility Guide](docs/model_compatibility.md) for GGUF model URLs, multimodal settings, and quant recommendations.
