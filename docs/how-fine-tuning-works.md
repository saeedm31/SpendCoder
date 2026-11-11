# Fine-tuning with Hugging Face, explained simply

A short guide to what you do in `notebooks/SpendCoder_finetuning.ipynb`, why each step exists, and how
companies use fine-tuned models in production.

## 1. What fine-tuning is

A **pretrained model** (BERT, RoBERTa, DistilRoBERTa, ...) has read a huge amount of text and learned how
language works: which words go together, what a sentence means. It knows nothing about your company's spend
categories.

**Fine-tuning** = take that pretrained model, add a small layer for your task (here: one output per spend
category), and train the whole thing a little more on your own labelled examples. The model keeps its language
knowledge and learns your task on top.

Analogy: hiring an experienced accountant (pretrained) and training them for a week on your chart of accounts
(fine-tuning), instead of teaching someone to read from zero (training from scratch).

Why it works with little data: the hard part, understanding language, is already done. A few thousand labelled
lines are enough to learn the mapping to categories.

## 2. The pieces of Hugging Face

| Piece | What it does | In the notebook |
|---|---|---|
| **Hub** | A website with thousands of pretrained models and datasets; also where you publish yours | `distilroberta-base`, your `spendcoder-distilroberta` |
| **Tokenizer** | Splits text into tokens and maps them to ids the model knows | `AutoTokenizer.from_pretrained(...)` |
| **Model** | The neural network; `AutoModelForSequenceClassification` adds the classification head | `num_labels=14`, `id2label` |
| **Datasets** | Fast tables that tokenize once with `.map(...)` | `Dataset.from_pandas(...)` |
| **Trainer** | The training loop: batches, loss, back-propagation, evaluation, checkpoints, early stopping | `Trainer(...)`, `trainer.train()` |
| **TrainingArguments** | The settings: learning rate, epochs, batch size, mixed precision, when to evaluate | see below |

## 3. The steps, and why each one matters

1. **Data with a time split.** Train on the past, validate on the next period, test on the newest. This is how the
   model will be used, so it is the honest test. Remove test lines that also appear in training, otherwise you
   measure memory, not understanding.
2. **Baseline first.** TF-IDF + logistic regression trains in seconds. If the transformer does not clearly beat
   it, the simpler model wins: cheaper, faster, easier to explain.
3. **Tokenize.** Check how long your texts are in tokens and pick `max_length` (64 here). Longer texts are cut.
4. **Load the model + head.** The warning "some weights are newly initialized" is the new head. Expected.
5. **Training arguments** (the ones to know):
   - **Learning rate** (`5e-5`): small steps, so the pretrained knowledge is adjusted, not destroyed. The most
     important setting; typical range 1e-5 to 1e-4.
   - **Epochs** (2 to 5): passes over the data. Too many = overfitting (great on training, worse on new data).
   - **Batch size** (16 to 64): lines per step; limited by GPU memory.
   - **Warmup** (about 6% of steps): starts with a tiny learning rate for stability.
   - **Weight decay** (0.01): light regularisation.
   - **Evaluate every epoch, keep the best** (`load_best_model_at_end`, `metric_for_best_model`), plus **early
     stopping**: training stops when validation stops improving.
   - **fp16** on a GPU: half-precision numbers, about twice as fast.
   - **Class imbalance:** if some categories have far fewer examples, weight the loss so mistakes on them cost
     more (here: a `Trainer` subclass that overrides `compute_loss`). Otherwise macro F1 suffers.
6. **Train.** Each step: forward pass, loss (cross-entropy: how wrong was it), backward pass (gradients), optimizer
   step (AdamW). Watch training loss go down and validation scores go up.
7. **Evaluate on the test set once.** Accuracy, macro F1 (every category counts equally), per-category scores,
   and real errors. Error analysis often shows unclear labels or texts too vague for anyone.
8. **Turn it into a decision rule.** Pick a confidence threshold on validation (for example: automatic codings
   must be 95% right). Above it: automatic. Below it: a person decides, with the top 3 suggestions.
9. **Package.** Save the model and tokenizer, write a model card, export to ONNX for light serving, publish to the
   Hub with a version.

## 4. Fine-tuning, prompting an LLM, or RAG: which one when?

| Approach | Best when | Trade-offs |
|---|---|---|
| **Classic ML** (TF-IDF + logistic regression) | Short texts, clear labels, need speed and simplicity | Weaker on new wording and synonyms |
| **Fine-tuned small transformer** (this project) | A fixed task (classify, extract) with thousands of labelled examples and high volume | Needs labelled data and a GPU for training; then cheap, fast, private, consistent |
| **Prompting a large LLM** (GPT, Claude) | Few or no labels, changing tasks, reasoning, generation | Per-call cost, slower, data leaves the company unless self-hosted, outputs can vary |
| **RAG** | Answers must come from your documents | Retrieval quality is the bottleneck; not a classifier |
| **Fine-tuning an LLM** (LoRA / PEFT) | A generative task needs a special style or format | More compute; usually tried after prompting and RAG |

A common production pattern: start with an LLM to label data or prove value, then fine-tune a small model for
the high-volume task, and keep the LLM for the rare hard cases.

## 5. Where companies use this

- **Finance and accounting:** code invoice and expense lines to GL accounts or cost categories, route accounts
  payable emails, flag unusual journal entries.
- **Procurement:** classify spend for analytics and savings projects, map supplier descriptions to a catalogue.
- **Customer service and IT:** route tickets to the right team, detect urgency.
- **Maintenance and HSE:** classify work orders and incident reports by failure type or risk.
- **Compliance:** classify contract clauses, detect personal data.

Why a fine-tuned small model there: thousands to millions of items a day, same task every time, data must stay
inside the company, and each prediction must be fast, cheap and auditable.

## 6. In production

```
new invoice line -> API (FastAPI, Docker) -> ONNX model -> category + confidence
                                                         |-- confidence >= threshold -> posted automatically
                                                         |-- below threshold          -> person reviews (top 3 shown)
person's decision -> stored as a new label -> next fine-tuning round
```

- **Serving:** ONNX + onnxruntime on CPU: small image, no GPU, milliseconds per line, batch endpoint.
- **Versioning:** each model version on the Hub (or a private registry) with its model card and test scores.
- **Monitoring:** share of lines sent to review, category mix, and agreement with reviewers. If these drift,
  wording or suppliers changed.
- **Retraining:** reviewers' corrections become new training data; retrain on a schedule, and promote the new
  model only if it beats the current one on the same held-out data.
- **Governance:** human review for low confidence, audit log of every suggestion and decision, no automatic
  posting above a value limit.
- **On a cloud platform:** the same steps map to managed services. On AWS: SageMaker training jobs for
  fine-tuning (GPU only while training), S3 for data and model versions, ECS Fargate or a SageMaker endpoint for
  the API, CloudWatch for monitoring. This repo includes that setup in Terraform ([AWS option](aws.md)).

## 7. Check your understanding

After working through the notebook, you will understand:
- the difference between pretraining and fine-tuning, and why fine-tuning needs little data
- what the tokenizer does, and why it must match the model
- learning rate, epochs, batch size, warmup, early stopping, fp16, class-weighted loss
- how to read the training log: if validation is still improving at the last epoch, train longer
- why a baseline first, and why a time-based split
- accuracy vs macro F1, and why macro F1 matters for small categories
- how a confidence threshold turns a model into a safe business process
- why ONNX for serving, and what a model card contains
- when you would fine-tune a small model instead of prompting an LLM, and the other way round
