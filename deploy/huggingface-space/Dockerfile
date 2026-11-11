# Hugging Face Docker Space: builds the API + web page from the GitHub repo and loads the model from the Hub.
# Copy this file and README.md into a new Space (SDK: Docker). To pick up new code: Settings -> Factory rebuild.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
RUN git clone --depth 1 https://github.com/saeedm31/SpendCoder.git . \
 && pip install --no-cache-dir --user onnxruntime tokenizers huggingface_hub fastapi uvicorn numpy pandas requests
ENV MODEL_REPO=Saeedhoj/spendcoder-distilroberta
EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
