"""Production inference: ONNX model + tokenizer, no PyTorch needed.

The decision rule is where the business logic lives: a line is coded automatically only if the model is
confident enough; otherwise it goes to a person with the top suggestions. The threshold is chosen on the
validation year so that automatic codings reach a target accuracy (see metrics.threshold_for_accuracy).
"""

import json
import os

import numpy as np

from .data import clean_text

FILES = ["model.onnx", "tokenizer.json", "spendcoder.json"]


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decide(text, probs, labels, threshold, top_k=3):
    """One line's answer: the category, how sure, whether a person must review, and the top suggestions."""
    order = np.argsort(probs)[::-1][:top_k]
    confidence = float(probs[order[0]])
    return {
        "text": text,
        "category": labels[order[0]],
        "confidence": round(confidence, 4),
        "needs_review": confidence < threshold,
        "suggestions": [{"category": labels[i], "confidence": round(float(probs[i]), 4)} for i in order],
    }


class Predictor:
    def __init__(self, tokenizer, session, labels, threshold=0.9, batch_size=32):
        self.tokenizer, self.session, self.labels, self.threshold = tokenizer, session, list(labels), threshold
        self.batch_size = batch_size     # 1 for the int8 model: its number scale is set per batch, so a line's
                                         # answer would otherwise depend on the other lines in the same request
        self.input_names = [i.name for i in session.get_inputs()]

    @classmethod
    def load(cls, model_dir, threshold=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(model_dir, "spendcoder.json")) as f:
            meta = json.load(f)
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_truncation(max_length=meta.get("max_length", 64))
        tok.enable_padding(pad_id=meta.get("pad_id", 1), pad_token=meta.get("pad_token", "<pad>"))
        session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
        obj = cls(tok, session, meta["labels"], threshold if threshold is not None else meta.get("threshold", 0.9))
        obj.meta = meta.get("metrics", {})
        return obj

    @classmethod
    def from_hub(cls, repo_id, revision=None, token=None, threshold=None, subfolder="onnx"):
        """Download an ONNX folder from a Hugging Face Hub model repo and load it.

        subfolder "onnx" is the full-precision model; "onnx-int8" is the 4x smaller int8 version for small servers.
        """
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, revision=revision, token=token, allow_patterns=[f"{subfolder}/*"])
        return cls.load(os.path.join(path, subfolder), threshold=threshold)

    def probs(self, texts, batch_size=None):
        """Category probabilities. Lines go through the model in small batches, so memory stays low on a
        small server even for a 500-line request."""
        if not texts:
            return np.zeros((0, len(self.labels)))
        batch_size = batch_size or self.batch_size
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch([clean_text(t) or " " for t in texts[i:i + batch_size]])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            logits = self.session.run(None, {n: feeds[n] for n in self.input_names if n in feeds})[0]
            out.append(softmax(np.asarray(logits, dtype=np.float64)))
        return np.concatenate(out)

    def classify(self, texts):
        return [decide(t, p, self.labels, self.threshold) for t, p in zip(texts, self.probs(texts))]
