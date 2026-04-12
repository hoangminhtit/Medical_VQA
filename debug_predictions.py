import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, T5Tokenizer
from collections import Counter

from config import get_args
from model import MedicalVQAModel
from dataset import MedicalVQADataset
from utils import load_checkpoint
from hf_runtime import configure_hf_runtime


def analyze_predictions(args, num_samples: int = 200):
    """Generate predictions and save to CSV for manual inspection.
    
    Args:
        args: CLI arguments
        num_samples: Number of test samples to analyze (default 200)
    """
    configure_hf_runtime(args)
    device = args.device

    # ── Model ──────────────────────────────────────────────────────
    model = MedicalVQAModel(
        encoder_dim=args.encoder_dim,
        vocab_size=args.vocab_size,
        max_answer_len=args.max_answer_len
    ).to(device)

    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # ── Tokenizer ──────────────────────────────────────────────────
    question_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # ── Dataset ────────────────────────────────────────────────────
    ds = load_dataset(args.dataset)
    test_split = ds.get("test") or ds.get("validation") or ds.get("val")
    if test_split is None:
        raise ValueError(
            f"Dataset '{args.dataset}' has no 'test', 'validation', or 'val' split. "
            f"Available splits: {list(ds.keys())}"
        )
    
    # Limit to first N samples for analysis
    limited_data = test_split.select(range(min(num_samples, len(test_split))))
    test_ds = MedicalVQADataset(limited_data)

    test_loader = DataLoader(
        test_ds, batch_size=8, num_workers=0  # Small batch, no workers for debug
    )
    print(f"Analyzing {len(test_ds)} samples...")

    # ── Collect Predictions ────────────────────────────────────────
    results = []
    answer_types = Counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            yn_labels = batch["yesno"].cpu()
            is_yn = batch["is_yesno"]
            gen_labels = batch["answer"]
            
            # Get model outputs
            yesno_logits, _ = model(images, input_ids, mask, generate_text=False)  # Training mode for Y/N
            
            # For text generation, use generation mode
            _, generated_ids = model(images, input_ids, mask, generate_text=True)  # Generation mode
            
            # Decode predictions  
            yn_preds = (yesno_logits.cpu() > 0).long().view(-1)
            
            # Decode to text
            questions = question_tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True)
            gt_answers = t5_tokenizer.batch_decode(gen_labels.cpu(), skip_special_tokens=True)
            pred_answers = t5_tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            
            # Process each sample
            for i in range(len(questions)):
                question = questions[i].strip()
                gt_answer = gt_answers[i].strip()
                pred_answer = pred_answers[i].strip()
                is_yesno = is_yn[i].item()
                
                # For yes/no questions, use classification head
                if is_yesno:
                    pred_final = "yes" if yn_preds[i] == 1 else "no"
                else:
                    pred_final = pred_answer
                
                # Count answer types for analysis
                answer_types[gt_answer.lower()] += 1
                
                results.append({
                    "question": question,
                    "ground_truth": gt_answer,
                    "predicted": pred_final,
                    "is_yesno": is_yesno,
                    "correct_yn": (yn_preds[i] == yn_labels[i]).item() if is_yesno else None,
                    "exact_match": (pred_final.lower() == gt_answer.lower()),
                })

    # ── Save Results ───────────────────────────────────────────────
    df = pd.DataFrame(results)
    output_path = f"predictions_analysis_{num_samples}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved predictions to: {output_path}")

    # ── Analysis Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    total = len(df)
    yesno_samples = df[df['is_yesno'] == True]
    open_samples = df[df['is_yesno'] == False]
    
    print(f"Total samples: {total}")
    print(f"Yes/No questions: {len(yesno_samples)} ({len(yesno_samples)/total*100:.1f}%)")
    print(f"Open-ended questions: {len(open_samples)} ({len(open_samples)/total*100:.1f}%)")
    
    if len(yesno_samples) > 0:
        yn_acc = yesno_samples['correct_yn'].mean()
        print(f"Yes/No accuracy: {yn_acc:.3f}")
    
    if len(open_samples) > 0:
        open_exact = open_samples['exact_match'].mean()
        print(f"Open-ended exact match: {open_exact:.3f}")
    
    # Show most common ground truth answers
    print(f"\nTop 15 most frequent answers:")
    for answer, count in answer_types.most_common(15):
        print(f"  {answer:20s} {count:3d} ({count/total*100:.1f}%)")
    
    # Show some examples of incorrect open-ended predictions
    print(f"\nSample incorrect open-ended predictions:")
    incorrect_open = df[(df['is_yesno'] == False) & (df['exact_match'] == False)][:10]
    for _, row in incorrect_open.iterrows():
        print(f"  Q: {row['question'][:60]}...")
        print(f"  GT: {row['ground_truth']}")
        print(f"  Pred: {row['predicted']}")
        print()

    return df


if __name__ == "__main__":
    import argparse
    
    # Use a simple parser for this debug script
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/kaggle/working/Medical_VQA/checkpoint.pt")
    parser.add_argument("--dataset", default="flaviagiammarino/path-vqa")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=32128)
    parser.add_argument("--max_answer_len", type=int, default=16)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_offline", action="store_true")
    parser.add_argument("--hf_timeout", type=int, default=120)
    parser.add_argument("--show_hf_warnings", action="store_true")
    parser.add_argument("--num_samples", type=int, default=200, 
                       help="Number of test samples to analyze")
    
    args = parser.parse_args()
    analyze_predictions(args, args.num_samples)