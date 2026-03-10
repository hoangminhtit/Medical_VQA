# Medical VQA — BioMedCLIP + Question-Aware Dual Gating + Phi-3 Mini

A Medical Visual Question Answering system that combines a frozen **BioMedCLIP** dual encoder with a novel **Question-Aware Dual Gating** module and a LoRA fine-tuned **Phi-3 Mini** decoder. Evaluated on **PathVQA** and **VQA-Med 2019**.

---

## Architecture

```
Image + Question
      ↓
BioMedCLIP Encoder  (frozen)
  ├─ Vision Encoder (ViT-B/16)  →  V ∈ R^{N×d}
  └─ Text Encoder (PubMedBERT)  →  Q ∈ R^{L×d}
      ↓
Question-Aware Dual Gating
  ├─ Gate 1 (Text → Image):  V' = V + σ(W₁·q)  ⊙  CrossAttn(V, Q)
  └─ Gate 2 (Image → Text):  Q' = Q + σ(W₂·v_cls) ⊙ CrossAttn(Q, V)
      ↓
Feature Fusion  →  concat(V', Q')
      ↓
Input Projection  (768 → 3072)
      ↓
Phi-3 Mini  +  LoRA  (fine-tuned)
      ↓
Two Output Heads
  ├─ Yes/No Classification Head  →  BCE Loss
  └─ Generative Answer Head      →  Cross-Entropy Loss
```

---

## Project Structure

```
Medical_VQA/
├── config.py              # CLI argument parser (get_args)
├── dataset.py             # MedicalVQADataset — HuggingFace & local images
├── data_processing.py     # Exploratory data loading / visualisation
├── dual_gating_attention.py  # Question-Aware Dual Gating Module
├── evaluate.py            # Evaluation: Y/N accuracy, BLEU-1, Exact Match
├── feature_extraction.py  # BioMedCLIPEncoder (frozen)
├── logger.py              # Shared file + console logger
├── loss.py                # Weighted BCE + CE loss
├── model.py               # MedicalVQAModel (full pipeline)
├── train.py               # Training loop
├── test.py                # Test-set evaluation
├── utils.py               # Checkpoint save / load
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
torchvision
transformers
timm
peft
datasets
nltk
```

> **Note:** NLTK punkt data is required for BLEU scoring. Run once:
> ```python
> import nltk; nltk.download("punkt")
> ```

---

## Datasets

| Dataset | HuggingFace ID | Split used |
|---------|----------------|------------|
| PathVQA | `flaviagiammarino/path-vqa` | train / val / test |
| VQA-Med 2019 | `flaviagiammarino/vqa-rad` | train / val / test |

Datasets are downloaded automatically by the `datasets` library on first run.

---

## Training

### Local

```bash
python train.py \
    --dataset    flaviagiammarino/path-vqa \
    --checkpoint checkpoint.pt \
    --log_dir    logs \
    --log_name   pathvqa_run1 \
    --batch_size 16 \
    --epochs     10 \
    --lr         1e-4 \
    --device     cuda
```

### Kaggle

```bash
python train.py \
    --dataset    flaviagiammarino/path-vqa \
    --checkpoint /kaggle/working/checkpoint.pt \
    --log_dir    /kaggle/working/logs \
    --log_name   pathvqa_run1 \
    --batch_size 32 \
    --epochs     20 \
    --device     cuda
```

The best checkpoint (highest validation Yes/No accuracy) is saved automatically.

---

## Evaluation (Test Set)

Pass the **same `--log_name`** as training to append test results into the same log file.

```bash
python test.py \
    --checkpoint /kaggle/working/checkpoint.pt \
    --log_dir    /kaggle/working/logs \
    --log_name   pathvqa_run1 \
    --dataset    flaviagiammarino/path-vqa \
    --device     cuda
```

### Metrics

| Question type | Metric |
|---------------|--------|
| Yes / No | Accuracy |
| Open-ended | BLEU-1, Exact Match |

---

## All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `flaviagiammarino/path-vqa` | HuggingFace dataset ID |
| `--checkpoint` | `checkpoint.pt` | Path to save/load checkpoint |
| `--log_dir` | `logs` | Directory for log files |
| `--log_name` | auto timestamp | Log file base name (no extension) |
| `--batch_size` | `16` | Batch size |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--num_workers` | `4` | DataLoader workers |
| `--encoder_dim` | `768` | BioMedCLIP output dimension |
| `--vocab_size` | `30522` | Vocabulary size |
| `--max_answer_len` | `16` | Max generated answer token length |
| `--loss_alpha` | `0.5` | Weight for Yes/No BCE loss |
| `--loss_beta` | `0.5` | Weight for generative CE loss |
| `--device` | auto | `cuda` or `cpu` |

---

## Logging

All runs log to both the **console** and a **file** under `--log_dir`.  
Using the same `--log_name` for train and test appends both into one file:

```
logs/
└── pathvqa_run1.log   ← training epochs + test results in one file
```

Log format:
```
2026-03-10 12:00:00 | INFO     | Training started
2026-03-10 12:00:01 | INFO     | Dataset     : flaviagiammarino/path-vqa
2026-03-10 12:00:01 | INFO     | Trainable params : 12,450,816 / 3,871,200,256
2026-03-10 12:05:32 | INFO     | Epoch 01/20 | Loss 0.6821 | Val Y/N Acc 0.7134 | Val BLEU 0.2310
...
2026-03-10 14:22:11 | INFO     | Test Set Results
2026-03-10 14:22:11 | INFO     | Yes/No  Accuracy : 0.8021  (3208/4000)
2026-03-10 14:22:11 | INFO     | Open    BLEU-1   : 0.3140
2026-03-10 14:22:11 | INFO     | Open    Exact    : 0.2876  (1150/4000)
```

---

## Model Details

| Component | Details |
|-----------|---------|
| Vision encoder | ViT-B/16 (via BioMedCLIP) — **frozen** |
| Text encoder | PubMedBERT (via BioMedCLIP) — **frozen** |
| Dual Gating | 2× CrossAttention (8 heads, dim=768) |
| Input projection | Linear(768 → 3072) |
| Decoder | Phi-3 Mini 4K + LoRA (r=16, α=32) |
| Yes/No head | Linear(3072 → 1) + BCEWithLogitsLoss |
| Generative head | Linear(3072 → vocab\_size) + CrossEntropyLoss |
| Optimizer | AdamW + CosineAnnealingLR |
