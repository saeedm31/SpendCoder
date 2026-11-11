"""SpendCoder API: classify purchase lines with the fine-tuned model (ONNX, CPU).

    MODEL_DIR=models/onnx uvicorn serve.app:app --port 8000
    # or pull the model from the Hugging Face Hub at start-up:
    MODEL_REPO=<user>/spendcoder-distilroberta uvicorn serve.app:app --port 8000
    # or on AWS (ECS Fargate), from the promoted model in S3:
    MODEL_S3_BUCKET=<bucket> uvicorn serve.app:app --port 8000

POST /classify  {"lines": ["ANNUAL MAINTENANCE OF HV SWITCHGEAR", "DELL LATITUDE LAPTOPS"]}
GET  /health
"""

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spendcoder.predictor import Predictor

MAX_LINES = 500
MAX_CHARS = 500

app = FastAPI(title="SpendCoder", version="1.0")


class ClassifyRequest(BaseModel):
    lines: list[str] = Field(..., min_length=1, max_length=MAX_LINES,
                             description="Purchase or invoice line descriptions")


@lru_cache(maxsize=1)
def predictor():
    if os.environ.get("MODEL_S3_BUCKET"):      # AWS: the model promoted to s3://<bucket>/models/active/onnx/
        import boto3

        from spendcoder import aws
        path = aws.download_active(boto3.client("s3"), os.environ["MODEL_S3_BUCKET"], "/tmp/spendcoder-model")
        return Predictor.load(path)
    if os.environ.get("MODEL_REPO"):
        return Predictor.from_hub(os.environ["MODEL_REPO"], revision=os.environ.get("MODEL_REVISION"),
                                  token=os.environ.get("HF_TOKEN"))
    return Predictor.load(os.environ.get("MODEL_DIR", "models/onnx"))


@app.get("/health")
def health():
    p = predictor()
    return {"ok": True, "categories": len(p.labels), "review_threshold": p.threshold}


@app.post("/classify")
def classify(req: ClassifyRequest):
    if any(len(line) > MAX_CHARS for line in req.lines):
        raise HTTPException(status_code=400, detail=f"Each line must be at most {MAX_CHARS} characters")
    results = predictor().classify(req.lines)
    return {"results": results,
            "summary": {"lines": len(results), "needs_review": sum(r["needs_review"] for r in results)}}
