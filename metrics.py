"""
Advanced evaluation metrics for Medical VQA.
Includes comprehensive BLEU score evaluation adapted from your training code.
"""

import torch
import math
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, brevity_penalty
from typing import List, Dict, Tuple


class BLEUEvaluator:
    """Comprehensive BLEU score evaluator for medical VQA."""
    
    def __init__(self):
        self.smoother = SmoothingFunction()
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.all_references = []
        self.all_hypotheses = []
        self.total_samples = 0
    
    def add_batch(self, references: List[str], hypotheses: List[str]):
        """Add a batch of predictions for evaluation.
        
        Args:
            references: List of ground truth answers
            hypotheses: List of predicted answers
        """
        # Token-level BLEU expects token lists, not raw strings.
        clean_refs = [ref.strip().lower().split() for ref in references]
        clean_hyps = [hyp.strip().lower().split() for hyp in hypotheses]

        self.all_references.extend([[ref] for ref in clean_refs])
        self.all_hypotheses.extend(clean_hyps)
        self.total_samples += len(references)
    
    def compute_bleu_scores(self) -> Dict[str, float]:
        """Compute comprehensive BLEU scores.
        
        Returns:
            Dictionary with BLEU-1,2,3,4 and composite BLEU score
        """
        if not self.all_references or not self.all_hypotheses:
            return {
                "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, 
                "bleu_composite": 0.0, "brevity_penalty": 1.0
            }
        
        # Calculate individual BLEU scores
        bleu1 = corpus_bleu(
            self.all_references, self.all_hypotheses,
            weights=(1.0, 0, 0, 0), 
            smoothing_function=self.smoother.method1
        )
        bleu2 = corpus_bleu(
            self.all_references, self.all_hypotheses,
            weights=(0.5, 0.5, 0, 0), 
            smoothing_function=self.smoother.method1
        )
        bleu3 = corpus_bleu(
            self.all_references, self.all_hypotheses,
            weights=(1/3, 1/3, 1/3, 0), 
            smoothing_function=self.smoother.method1
        )
        bleu4 = corpus_bleu(
            self.all_references, self.all_hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25), 
            smoothing_function=self.smoother.method1
        )
        
        # Calculate brevity penalty (reference length vs candidate length)
        c = sum(len(hyp) for hyp in self.all_hypotheses)      # Candidate length
        r = sum(len(ref[0]) for ref in self.all_references)   # Reference length
        bp = brevity_penalty(r, c)

        # Composite BLEU score (match sample code logic)
        bleu_scores = [bleu1, bleu2, bleu3, bleu4]
        valid_bleu_scores = [score for score in bleu_scores if score > 0]
        if valid_bleu_scores:
            composite_bleu = bp * math.exp(
                sum(math.log(score) for score in valid_bleu_scores) / len(valid_bleu_scores)
            )
        else:
            composite_bleu = 0.0
        
        return {
            "bleu1": bleu1,
            "bleu2": bleu2, 
            "bleu3": bleu3,
            "bleu4": bleu4,
            "bleu_composite": composite_bleu,
            "brevity_penalty": bp,
            "num_samples": self.total_samples
        }


class ExactMatchEvaluator:
    """Exact match accuracy evaluator."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def add_batch(self, references: List[str], hypotheses: List[str]):
        """Add batch for exact match evaluation."""
        for ref, hyp in zip(references, hypotheses):
            self.total += 1
            if ref.strip().lower() == hyp.strip().lower():
                self.correct += 1
    
    def compute_accuracy(self) -> float:
        """Compute exact match accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0


class MedicalVQAMetrics:
    """Complete metrics suite for Medical VQA evaluation."""
    
    def __init__(self):
        self.bleu_evaluator = BLEUEvaluator()
        self.exact_match_evaluator = ExactMatchEvaluator()
        self.yesno_correct = 0
        self.yesno_total = 0
    
    def reset(self):
        """Reset all metrics."""
        self.bleu_evaluator.reset()
        self.exact_match_evaluator.reset()
        self.yesno_correct = 0
        self.yesno_total = 0
    
    def add_yesno_batch(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Add Yes/No classification results.
        
        Args:
            predictions: (B,) tensor of predicted labels (0/1)
            labels: (B,) tensor of ground truth labels (0/1) 
        """
        self.yesno_correct += (predictions == labels).sum().item()
        self.yesno_total += labels.size(0)
    
    def add_openended_batch(self, references: List[str], hypotheses: List[str]):
        """Add open-ended generation results.
        
        Args:
            references: List of ground truth answers
            hypotheses: List of predicted answers
        """
        self.bleu_evaluator.add_batch(references, hypotheses)
        self.exact_match_evaluator.add_batch(references, hypotheses)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Yes/No accuracy
        yesno_acc = self.yesno_correct / self.yesno_total if self.yesno_total > 0 else None
        
        # Open-ended metrics
        bleu_metrics = self.bleu_evaluator.compute_bleu_scores()
        exact_acc = self.exact_match_evaluator.compute_accuracy()
        
        return {
            "yesno_accuracy": yesno_acc,
            "open_exact_match": exact_acc,
            **bleu_metrics  # Includes bleu1, bleu2, bleu3, bleu4, bleu_composite, brevity_penalty
        }
    
    def print_metrics(self, metrics: Dict[str, float], prefix: str = "Evaluation"):
        """Pretty print metrics."""
        print("=" * 70)
        print(f"{prefix} Results")
        print("=" * 70)
        
        # Yes/No accuracy
        if metrics["yesno_accuracy"] is not None:
            print(f"  Yes/No Accuracy      : {metrics['yesno_accuracy']:.4f}")
        else:
            print(f"  Yes/No Accuracy      : N/A")
        
        # Open-ended metrics
        print(f"  Open-ended Exact     : {metrics['open_exact_match']:.4f}")
        print("-" * 70)
        print(f"  BLEU-1               : {metrics['bleu1']:.4f}")
        print(f"  BLEU-2               : {metrics['bleu2']:.4f}")
        print(f"  BLEU-3               : {metrics['bleu3']:.4f}")
        print(f"  BLEU-4               : {metrics['bleu4']:.4f}")
        print(f"  BLEU Composite       : {metrics['bleu_composite']:.4f}")
        print(f"  Brevity Penalty      : {metrics['brevity_penalty']:.4f}")
        
        if "num_samples" in metrics:
            print(f"  Evaluated Samples    : {metrics['num_samples']}")
        
        print("=" * 70)


# Convenience function for quick evaluation
def evaluate_medical_vqa(
    model, dataloader, device, 
    t5_tokenizer, biomedclip_tokenizer=None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Complete Medical VQA evaluation with comprehensive metrics.
    
    Args:
        model: Medical VQA model
        dataloader: Test/validation dataloader
        device: torch device
        t5_tokenizer: T5 tokenizer for decoding
        biomedclip_tokenizer: BioMedCLIP tokenizer (if needed for GT)
        verbose: Whether to print results
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    metrics = MedicalVQAMetrics()
    
    debug_printed = False

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            yn_labels = batch["yesno"].to(device)
            is_yn = batch["is_yesno"].to(device=device, dtype=torch.bool)
            gen_lbl = batch["answer"]  # CPU for decoding
            
            # Model predictions (single encoder/gating pass, eval_mode shares encoder)
            yesno_logits, generated_ids = model(images, input_ids, mask, eval_mode=True)
            
            # Yes/No evaluation
            if is_yn.any():
                yn_preds = (yesno_logits[is_yn] > 0).long().squeeze(-1)
                yn_gt = yn_labels[is_yn]
                metrics.add_yesno_batch(yn_preds, yn_gt)
            
            # Open-ended evaluation  
            open_mask = ~is_yn
            if open_mask.any():
                generated_ids_cpu = generated_ids.cpu()
                open_mask_cpu = open_mask.cpu()

                # Decode predictions
                pred_texts = t5_tokenizer.batch_decode(
                    generated_ids_cpu[open_mask_cpu], skip_special_tokens=True
                )

                # Decode ground truth
                # Replace -100 (ignore index set by dataset) back to pad_id before decoding
                gt_for_decode = gen_lbl[open_mask_cpu].clone()
                gt_for_decode[gt_for_decode == -100] = t5_tokenizer.pad_token_id
                gt_texts = t5_tokenizer.batch_decode(
                    gt_for_decode, skip_special_tokens=True
                )

                if verbose and not debug_printed and pred_texts:
                    print(f"[DEBUG] Sample prediction: '{pred_texts[0]}'")
                    print(f"[DEBUG] Sample ground truth: '{gt_texts[0]}'")
                    debug_printed = True

                metrics.add_openended_batch(gt_texts, pred_texts)
    
    # Compute final metrics
    results = metrics.compute_metrics()
    
    if verbose:
        metrics.print_metrics(results)
    
    return results