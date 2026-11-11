---
title: SpendCoder
emoji: 🧾
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fine-tuned transformer that codes purchase lines
models:
  - Saeedhoj/spendcoder-distilroberta
---

# SpendCoder

Paste purchase or invoice lines and get a spend category, the model's confidence, whether a person should review
it, and the top 3 suggestions. The same container is a REST API: `POST /classify`, `GET /health`, docs at `/docs`.

- Model: [Saeedhoj/spendcoder-distilroberta](https://huggingface.co/Saeedhoj/spendcoder-distilroberta)
  (DistilRoBERTa fine-tuned with the Hugging Face Trainer, served as ONNX on CPU)
- Code, notebook and results: [github.com/saeedm31/SpendCoder](https://github.com/saeedm31/SpendCoder)
