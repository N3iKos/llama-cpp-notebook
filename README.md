# llama-cpp-notebook

Notebook backend for running GGUF LLM/VLM models with prebuilt `llama.cpp`.

Targets:
- Google Colab T4 1x
- Kaggle T4 2x

Outputs:
- OpenAI-compatible endpoint: `/v1/chat/completions`
- Public tunnel using ngrok and/or Cloudflare Quick Tunnel

Notebooks:
- `notebooks/colab_t4_1x.ipynb`
- `notebooks/kaggle_t4_2x.ipynb`

Defaults:
- no model-name specific logic
- no forced image token override
- no forced thinking/reasoning override
