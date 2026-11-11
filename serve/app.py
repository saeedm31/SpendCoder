"""SpendCoder API and web page: classify purchase lines with the fine-tuned model (ONNX, CPU).

    MODEL_DIR=models/onnx uvicorn serve.app:app --port 8000
    # or pull the model from the Hugging Face Hub at start-up:
    MODEL_REPO=<user>/spendcoder-distilroberta uvicorn serve.app:app --port 8000
    # MODEL_SUBFOLDER=onnx-int8 picks the 4x smaller int8 model (for servers with little memory)
    # or on AWS (ECS Fargate), from the promoted model in S3:
    MODEL_S3_BUCKET=<bucket> uvicorn serve.app:app --port 8000

GET  /          web page: paste lines, see category, confidence, review flag and top 3
POST /classify  {"lines": ["ANNUAL MAINTENANCE OF HV SWITCHGEAR", "DELL LATITUDE LAPTOPS"]}
GET  /health    model status, categories, review threshold, test scores
GET  /docs      interactive API documentation (OpenAPI)
"""

import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from spendcoder.predictor import Predictor

MAX_LINES = 500
MAX_CHARS = 500
PAGE = Path(__file__).parent / "static" / "index.html"
_warmup = {"running": False, "error": None}


@lru_cache(maxsize=1)
def predictor():
    if os.environ.get("MODEL_S3_BUCKET"):      # AWS: the model promoted to s3://<bucket>/models/active/onnx/
        import boto3

        from spendcoder import aws
        path = aws.download_active(boto3.client("s3"), os.environ["MODEL_S3_BUCKET"], "/tmp/spendcoder-model")
        return Predictor.load(path)
    if os.environ.get("MODEL_REPO"):
        return Predictor.from_hub(os.environ["MODEL_REPO"], revision=os.environ.get("MODEL_REVISION"),
                                  token=os.environ.get("HF_TOKEN"),
                                  subfolder=os.environ.get("MODEL_SUBFOLDER", "onnx"))
    return Predictor.load(os.environ.get("MODEL_DIR", "models/onnx"))


def _warm_up():
    """Load the model in the background at start-up, so the page opens at once and the first call is fast."""
    try:
        predictor()
    except Exception as e:      # reported by /health; a later call retries
        _warmup["error"] = f"{type(e).__name__}: {e}"
    finally:
        _warmup["running"] = False


@asynccontextmanager
async def lifespan(app):
    if os.environ.get("WARMUP", "1") != "0":
        _warmup["running"] = True
        threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="SpendCoder", version="1.1", lifespan=lifespan,
              description="Codes purchase and invoice lines into 14 spend categories. Lines below the review "
                          "threshold should go to a person.")


class ClassifyRequest(BaseModel):
    lines: list[str] = Field(..., min_length=1, max_length=MAX_LINES,
                             description="Purchase or invoice line descriptions")


def _model_source():
    if os.environ.get("MODEL_S3_BUCKET"):
        return {"type": "s3", "name": os.environ["MODEL_S3_BUCKET"]}
    if os.environ.get("MODEL_REPO"):
        return {"type": "hub", "name": os.environ["MODEL_REPO"], "variant": os.environ.get("MODEL_SUBFOLDER", "onnx"),
                "url": f"https://huggingface.co/{os.environ['MODEL_REPO']}"}
    return {"type": "folder", "name": os.environ.get("MODEL_DIR", "models/onnx")}


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE, media_type="text/html")


@app.get("/health")
def health():
    if _warmup["running"]:
        return {"ok": False, "status": "loading", "model": _model_source()}
    try:
        p = predictor()
    except Exception as e:
        return {"ok": False, "status": "error", "error": f"{type(e).__name__}: {e}", "model": _model_source()}
    return {"ok": True, "status": "ready", "categories": len(p.labels), "labels": p.labels,
            "review_threshold": p.threshold, "metrics": getattr(p, "meta", {}), "model": _model_source(),
            "limits": {"lines": MAX_LINES, "chars": MAX_CHARS}}


@app.post("/classify")
def classify(req: ClassifyRequest):
    if any(len(line) > MAX_CHARS for line in req.lines):
        raise HTTPException(status_code=400, detail=f"Each line must be at most {MAX_CHARS} characters")
    if _warmup["running"]:
        raise HTTPException(status_code=503, detail="The model is still loading. Try again in a few seconds.")
    results = predictor().classify(req.lines)
    return {"results": results,
            "summary": {"lines": len(results), "needs_review": sum(r["needs_review"] for r in results)}}
