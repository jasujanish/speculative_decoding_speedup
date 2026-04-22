## 1. Setup

Create the Modal virtual environment from the repository root, then run the
workflow commands from `Learning-to-Draft-main`:

```bash
uv venv .venv/modal
source .venv/modal/bin/activate
uv pip install modal
modal setup
cd Learning-to-Draft-main
```

## 2. Download Models

```bash
modal run modal_qwen3.py::download_models --model-preset qwen3_8b
modal run modal_qwen3.py::download_models --model-preset qwen3_14b
```

Weights are cached in the `ltd-qwen3-models` Modal volume. These weights are shared by the LTD workflow and the supervised-depth workflow.
