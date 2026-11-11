"""Export the fine-tuned model to ONNX for light, fast CPU serving.

Why: the PyTorch model needs torch (about 1 GB installed) and a Python training stack. The ONNX file runs with
onnxruntime and the tokenizers library only (about 50 MB), starts fast and is quicker on CPU. The output
folder is everything the API and the demo need: model.onnx, tokenizer.json and spendcoder.json.
"""

import json
import os

import numpy as np

from .finetune import MAX_LENGTH
from .taxonomy import CATEGORIES


def export_onnx(model, tokenizer, out_dir, threshold=0.9, max_length=MAX_LENGTH, metrics=None):
    import torch

    os.makedirs(out_dir, exist_ok=True)
    model = model.float().cpu().eval()

    class LogitsOnly(torch.nn.Module):          # ONNX wants plain tensors in and out
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = LogitsOnly(model)
    sample = tokenizer(["ANNUAL MAINTENANCE OF HV SWITCHGEAR", "LAPTOPS"], padding=True, return_tensors="pt")
    args = (sample["input_ids"], sample["attention_mask"])
    path = os.path.join(out_dir, "model.onnx")
    names = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"])
    try:      # classic exporter
        torch.onnx.export(wrapped, args, path, dynamo=False, opset_version=17, **names,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch"}})
    except Exception as e:   # newer torch versions: the dynamo-based exporter (needs the onnxscript package)
        print(f"Classic ONNX export unavailable ({type(e).__name__}); using the dynamo exporter.")
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=512)
        torch.onnx.export(wrapped, args, path, dynamo=True, external_data=False, **names,
                          dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}))

    tokenizer.save_pretrained(out_dir)           # writes tokenizer.json (the fast tokenizer) and friends
    meta = {"labels": CATEGORIES, "max_length": max_length, "pad_id": tokenizer.pad_token_id,
            "pad_token": tokenizer.pad_token, "threshold": threshold, "metrics": metrics or {}}
    with open(os.path.join(out_dir, "spendcoder.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def check_parity(model, tokenizer, onnx_dir, texts):
    """The ONNX model must give the same answers as PyTorch. Returns the largest probability difference."""
    from .finetune import predict_proba
    from .predictor import Predictor

    torch_p = predict_proba(model, tokenizer, texts)
    onnx_p = Predictor.load(onnx_dir).probs(texts)
    return float(np.abs(torch_p - onnx_p).max())
