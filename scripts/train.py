"""Baseline, fine-tuning, evaluation, review threshold, ONNX export and (optionally) Hub upload.

    python scripts/train.py --baseline-only                  # seconds, no GPU or torch needed
    python scripts/train.py                                  # fine-tune (GPU recommended: about 10 min on a T4)
    python scripts/train.py --push <hf-user>/spendcoder-distilroberta   # also upload to the Hugging Face Hub
    python scripts/aws_train.py                              # or run it on AWS SageMaker (see docs/aws.md)

The Hub upload needs a token with write access: `huggingface-cli login` or the HF_TOKEN environment variable.
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spendcoder import data, metrics  # noqa: E402
from spendcoder.baseline import TfidfBaseline  # noqa: E402


def load(path):
    df = pd.read_csv(path, parse_dates=["start_date"])
    return data.split(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spend_lines.csv.gz")
    p.add_argument("--model", default="distilroberta-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", default="models/spendcoder")
    p.add_argument("--target-accuracy", type=float, default=0.95)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--push", default=None, help="Hugging Face Hub repo id, e.g. user/spendcoder-distilroberta")
    a = p.parse_args()

    train, val, test = load(a.data)
    print(f"train {len(train):,} · val {len(val):,} · test {len(test):,}")
    results = {"rows": {"train": len(train), "val": len(val), "test": len(test)}}

    base = TfidfBaseline().fit(train["text"], train["label"])
    results["baseline_test"] = metrics.scores(test["label"], base.predict_proba(test["text"]))
    print("Baseline (test):", results["baseline_test"])
    os.makedirs("reports", exist_ok=True)
    if a.baseline_only:
        json.dump(results, open("reports/baseline.json", "w"), indent=2)
        return

    from spendcoder import export, finetune
    from spendcoder.modelcard import model_card

    trainer = finetune.train(train, val, model_name=a.model, output_dir=a.out, epochs=a.epochs)
    model, tok = trainer.model, trainer.processing_class
    val_p = finetune.predict_proba(model, tok, val["text"].tolist())
    test_p = finetune.predict_proba(model, tok, test["text"].tolist())
    results["finetuned_val"] = metrics.scores(val["label"], val_p)
    results["finetuned_test"] = metrics.scores(test["label"], test_p)
    results["threshold"] = metrics.threshold_for_accuracy(val["label"], val_p, a.target_accuracy)
    results["automation_test"] = metrics.automation(
        test["label"], test_p, thresholds=sorted({0.5, 0.7, 0.9, results["threshold"]}))
    results["per_category_test"] = metrics.report(test["label"], test_p)
    print("Fine-tuned (test):", results["finetuned_test"])
    print("Review threshold:", results["threshold"])

    onnx_dir = os.path.join(a.out, "onnx")
    export.export_onnx(model, tok, onnx_dir, threshold=results["threshold"],
                       metrics={k: results[k] for k in ("baseline_test", "finetuned_test")})
    results["onnx_max_prob_diff"] = export.check_parity(model, tok, onnx_dir, test["text"].head(200).tolist())
    print("ONNX vs PyTorch, largest probability difference:", results["onnx_max_prob_diff"])
    json.dump(results, open("reports/finetuned.json", "w"), indent=2)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)   # kept with the model
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(model_card(results, base_model=a.model))
    for name in os.listdir(a.out):            # training checkpoints are not needed once the best model is saved
        if name.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(a.out, name), ignore_errors=True)

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, exist_ok=True)
        model.push_to_hub(a.push)
        tok.push_to_hub(a.push)
        api.upload_folder(repo_id=a.push, folder_path=onnx_dir, path_in_repo="onnx")
        api.upload_file(repo_id=a.push, path_or_fileobj=os.path.join(a.out, "README.md"), path_in_repo="README.md")
        print(f"Uploaded: https://huggingface.co/{a.push}")


if __name__ == "__main__":
    main()
