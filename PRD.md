# PRD — llama-cpp-notebook

Repository target: `github.com/N3iKos/llama-cpp-notebook`

## Objective

Build a compact notebook backend for serving GGUF text and multimodal models through `llama.cpp` without compiling from source. The project supports Google Colab T4 1x and Kaggle T4 2x.

## Platform targets

### Google Colab T4 1x

Notebook count: 1 primary notebook.

Cell count: 1–3 cells. The main setup cell uses Colab `#@param` fields.

Default GPU config:

```text
CUDA_VISIBLE_DEVICES=0
SPLIT_MODE=none
TENSOR_SPLIT=1
```

### Kaggle T4 2x

Notebook count: 1 primary notebook.

Cell count: 7–8 cells.

Default GPU config:

```text
CUDA_VISIBLE_DEVICES=0,1
SPLIT_MODE=row
FALLBACK_SPLIT_MODE=layer
TENSOR_SPLIT=1,1
```

## Requirements

- Install prebuilt CUDA `llama.cpp`.
- Download model GGUF and optional mmproj GGUF.
- Start `llama-server`.
- Warm up server.
- Create public tunnel links.
- Support ngrok and Cloudflare Quick Tunnel.
- Keep backend model-neutral.
- Use terminal-like live output in notebooks.
- Keep notebook cells short.
- Store long logic in Python modules.

## References

- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- ai-dock llama.cpp CUDA prebuilt: https://github.com/ai-dock/llama.cpp-cuda
- pyngrok docs: https://pyngrok.readthedocs.io/en/latest/
- Cloudflare Quick Tunnel docs: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- IPython `clear_output`: https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
