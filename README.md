# Medical VQA — BioMedCLIP + Question-Aware Dual Gating + T5

A Medical Visual Question Answering system that combines a frozen **BioMedCLIP** dual encoder with a **Question-Aware Dual Gating** module and a **T5-base** decoder for answer generation. Evaluated on **PathVQA** and **VQA-Med 2019 (vqa-rad proxy)**.

---

## Architecture

```
Image + Question
      ↓
BioMedCLIP Encoder (frozen)
  ├─ Vision Encoder (ViT-B/16)      → V ∈ R^{N×d}
  └─ Text Encoder (PubMedBERT)      → Q ∈ R^{L×d}
      ↓
Question-Aware Dual Gating
  ├─ Gate 1 (Text → Image): V' = V + σ(W1·q) ⊙ CrossAttn(V, Q)
  └─ Gate 2 (Image → Text): Q' = Q + σ(W2·v_cls) ⊙ CrossAttn(Q, V)
      ↓
Feature Fusion: concat(V', Q')
      ↓
T5 Decoder (T5ForConditionalGeneration)
      ↓
Two Output Heads
  ├─ Yes/No Classification Head  → BCEWithLogitsLoss (masked by is_yesno)
  └─ Generative Answer Head      → Token CE Loss (T5 logits)
```

---

## Project Structure

```
Medical_VQA/
├── config.py
├── dataset.py
├── data_processing.py
├── dual_gating_attention.py
├── evaluate.py
├── feature_extraction.py
├── logger.py
├── loss.py
├── metrics.py
├── model.py
├── test.py
├── train.py
├── utils.py
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
transformers>=4.40.0
timm
datasets
nltk
open-clip-torch
pandas
matplotlib
Pillow
```

Note: NLTK punkt data is required for BLEU scoring.

```python
import nltk
nltk.download("punkt")
```

---

## Run Without HF Token

You can run this project without `HF_TOKEN` for public models/datasets.

- Default behavior now hides unauthenticated HF warnings.
- Add `--hf_cache_dir ./hf_cache` to reuse local cache.
- Add `--hf_timeout 120` (or lower/higher) to control hub request timeout.
- Add `--hf_offline` to force fully offline mode (cache must already exist).
- Add `--show_hf_warnings` if you want to see HF hub warnings/logs.

Example:

```bash
python train.py \
  --dataset flaviagiammarino/path-vqa \
  --hf_cache_dir ./hf_cache \
  --hf_timeout 120
```

---

## Datasets

| Dataset | HuggingFace ID | Split used |
|---------|----------------|------------|
| PathVQA | `flaviagiammarino/path-vqa` | train / val / test |
| VQA-Med 2019 proxy | `flaviagiammarino/vqa-rad` | train / val / test |

If a split is missing, train/test scripts apply fallback logic (validation/val/test, or split from train when needed).

---

## Training

```bash
python train.py \
  --dataset flaviagiammarino/path-vqa \
  --checkpoint checkpoint.pt \
  --log_dir logs \
  --log_name pathvqa_run1 \
  --batch_size 16 \
  --epochs 10 \
  --lr 1e-4 \
  --device cuda
```

The best checkpoint is saved by highest validation Yes/No accuracy (strict improvement only).

---

## Evaluation (Test Set)

Use the same `--log_name` as training to append into one log file.

```bash
python test.py \
  --checkpoint checkpoint.pt \
  --log_dir logs \
  --log_name pathvqa_run1 \
  --dataset flaviagiammarino/path-vqa \
  --device cuda
```

### Metrics

| Question type | Metric |
|---------------|--------|
| Yes / No | Accuracy |
| Open-ended | Exact Match, BLEU-1/2/3/4, Composite BLEU, Brevity Penalty |

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `flaviagiammarino/path-vqa` | HuggingFace dataset ID |
| `--checkpoint` | `checkpoint.pt` | Save/load checkpoint path |
| `--log_dir` | `logs` | Directory for log files |
| `--log_name` | auto timestamp | Log file base name (without extension) |
| `--batch_size` | `16` | Batch size |
| `--epochs` | `10` | Number of epochs |
| `--lr` | `1e-4` | Learning rate |
| `--num_workers` | `4` | DataLoader workers |
| `--encoder_dim` | `768` | BioMedCLIP feature dim |
| `--vocab_size` | `32128` | T5-base vocabulary size |
| `--max_answer_len` | `16` | Max generated answer length |
| `--loss_alpha` | `1.0` | Yes/No BCE weight |
| `--loss_beta` | `0.5` | Generative CE weight |
| `--early_stopping` | `3` | Stop if val Y/N does not improve for N epochs (0 disables) |
| `--device` | auto | `cuda` or `cpu` |

---

## Model Details

| Component | Details |
|-----------|---------|
| Vision encoder | ViT-B/16 via BioMedCLIP (frozen) |
| Text encoder | PubMedBERT via BioMedCLIP (frozen) |
| Dual Gating | 2x cross-attention (8 heads, dim=768) |
| Decoder | T5-base (`T5ForConditionalGeneration`) |
| Input projection | Optional Linear(encoder_dim -> 768) if dim mismatch |
| Yes/No head | Linear(768 -> 1) + BCEWithLogitsLoss |
| Generative head | T5 LM head + CE loss |
| Optimizer | AdamW + CosineAnnealingLR |
