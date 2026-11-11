"""Model card for the Hugging Face Hub: what the model is, how it was tested, where it should not be used."""

from .taxonomy import CATEGORIES


def _row(name, m):
    return f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |"


def model_card(results, repo_url="https://github.com/saeedm31/SpendCoder", base_model="distilroberta-base"):
    t = results["threshold"]
    auto = next((r for r in results["automation_test"] if r["threshold"] == t), None)
    auto_line = (f"At the chosen review threshold ({t:.2f}), {auto['automated_pct']}% of test lines are coded "
                 f"automatically with {auto['accuracy_of_automated_pct']}% accuracy; the rest go to a person."
                 if auto else f"Review threshold: {t:.2f}.")
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    return f"""---
license: apache-2.0
base_model: {base_model}
language: en
pipeline_tag: text-classification
library_name: transformers
tags: [finance, accounting, spend-classification, procurement, onnx]
metrics: [accuracy, f1]
---

# SpendCoder: purchase line to spend category

Fine-tuned **{base_model}** that reads a short purchase or invoice line (for example
"ANNUAL MAINTENANCE OF HV SWITCHGEAR") and returns one of 14 spend categories, with a confidence score.
Low-confidence lines are meant to go to a person. Code, data preparation and tests: {repo_url}

## Results (test year, contracts the model never saw)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
{_row("TF-IDF + logistic regression (baseline)", results["baseline_test"])}
{_row("Fine-tuned " + base_model, results["finetuned_test"])}

{auto_line}

Train: {results["rows"]["train"]:,} lines (contracts starting before Oct 2023). Validation: {results["rows"]["val"]:,}
(Oct 2023 to Sep 2024). Test: {results["rows"]["test"]:,} (from Oct 2024). Test lines whose exact text appears in
training were removed.

## Categories
{cats}

## Data
US Department of Energy contract descriptions and their Product and Service Codes from
[USAspending.gov](https://www.usaspending.gov) (public domain), grouped into the 14 categories above.

## Use
Suggest a spend category for purchase and invoice lines in accounts payable or procurement; send lines below the
review threshold to a person. Not for automatic posting without review. Trained on one public-sector buyer's
wording: before using it in another company, check it on that company's labelled lines and fine-tune further.

## Files
- PyTorch / Transformers weights at the repo root: `AutoModelForSequenceClassification.from_pretrained(...)`
- `onnx/`: `model.onnx`, `tokenizer.json`, `spendcoder.json` (labels, review threshold) for CPU serving with
  onnxruntime, no PyTorch needed
"""
