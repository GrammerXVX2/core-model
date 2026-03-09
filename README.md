vLLM + Ollama-compat proxy

Overview
- vLLM serves OpenAI-compatible API on port 8000.
- Proxy exposes Ollama-compatible endpoints on port 11434.

Requirements
- Docker with NVIDIA runtime
- 2x RTX 5090
- vLLM nightly image (required for Qwen3.5 support before vLLM 0.17.0)

Setup
1) Get a Hugging Face token:
   - https://huggingface.co/settings/tokens
   - Create a "Read" token.
2) Copy .env.example to .env and set HF_TOKEN if required.

Run
- docker compose pull
- docker compose up

Endpoints
- POST http://localhost:11434/api/chat
- POST http://localhost:11434/api/generate

Notes
- Streaming is not implemented yet.
- Adjust --max-num-seqs and --gpu-memory-utilization in docker-compose.yml based on load.
- vLLM is pinned to `vllm/vllm-openai:nightly` for Qwen3.5 compatibility.
