# SpendCoder API: ONNX model on CPU, no PyTorch (image about 300 MB)
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir onnxruntime tokenizers huggingface_hub fastapi uvicorn numpy pandas requests boto3
COPY spendcoder/ spendcoder/
COPY serve/ serve/
# The model is loaded at start-up from one of:
#   MODEL_S3_BUCKET=<bucket>   the promoted model in S3 (AWS / ECS Fargate, see infra/terraform)
#   MODEL_REPO=<user>/<repo>   the Hugging Face Hub:  docker run -e MODEL_REPO=... -p 8000:8000 spendcoder
#   MODEL_DIR=models/onnx      a folder baked into or mounted in the container
ENV MODEL_DIR=models/onnx
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
