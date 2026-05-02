import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, T5Tokenizer
import torchvision.transforms as transforms


class MedicalVQADataset(Dataset):
    """Medical VQA dataset.

    Encoders changed:
      - Question tokenizer: BioBERT-base (replaces BioMedCLIP PubMedBERT)
      - Answer  tokenizer: T5-small (replaces T5-base)
    Image preprocessing: BLIP-2 normalisation.
    """

    # BLIP-2 / ViT standard normalisation (ImageNet stats)
    _MEAN = (0.48145466, 0.4578275,  0.40821073)
    _STD  = (0.26862954, 0.26130258, 0.27577711)

    # Tokenizer model IDs
    _QUESTION_TOKENIZER = "dmis-lab/biobert-base-cased-v1.2"
    _ANSWER_TOKENIZER   = "t5-small"

    def __init__(self, data, image_folder=None, max_answer_len: int = 16):
        """
        Args:
            data           : list of dicts OR HuggingFace Dataset.
                             Expected keys: 'image', 'question', 'answer'.
                             'image' may be a filename (str) or a PIL Image.
            image_folder   : base folder when 'image' is a filename.
                             Pass None (default) when using HuggingFace datasets
                             where 'image' is already a PIL Image.
            max_answer_len : maximum length for T5 answer tokens.
        """
        self.data           = data
        self.image_folder   = image_folder
        self.max_answer_len = max_answer_len

        # BioBERT tokenizer for questions → matches TextEncoder in feature_extraction.py
        self.question_tokenizer = AutoTokenizer.from_pretrained(self._QUESTION_TOKENIZER)

        # T5-small tokenizer for answers → matches T5Decoder in model.py
        self.answer_tokenizer = T5Tokenizer.from_pretrained(self._ANSWER_TOKENIZER)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD)
        ])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def is_yesno(answer: str) -> bool:
        return answer.strip().lower() in {"yes", "no"}

    def __getitem__(self, idx):
        sample = self.data[idx]

        # ── Image ───────────────────────────────────────────────────────
        raw = sample["image"]
        if isinstance(raw, str):
            raw = Image.open(f"{self.image_folder}/{raw}")
        image = self.transform(raw.convert("RGB"))

        # ── Tokenise question (BioBERT) ──────────────────────────────────
        question = sample["question"]
        q_tokens = self.question_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=64,          # BioBERT handles up to 512; 64 is plenty for VQA
            return_tensors="pt"
        )
        input_ids = q_tokens["input_ids"].squeeze(0)       # (L,)
        mask      = q_tokens["attention_mask"].squeeze(0)  # (L,)

        # ── Answer labels (T5-small tokenizer) ──────────────────────────
        answer  = sample["answer"]
        yn_flag = self.is_yesno(answer)
        yn_label = (1 if answer.strip().lower() == "yes" else 0) if yn_flag else -1

        gen_tokens = self.answer_tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_answer_len,
            return_tensors="pt"
        )
        gen_label = gen_tokens["input_ids"].squeeze(0)  # (max_answer_len,)

        # Replace padding with -100 (PyTorch ignore_index convention)
        pad_id    = self.answer_tokenizer.pad_token_id   # 0 for T5
        gen_label = gen_label.masked_fill(gen_label == pad_id, -100)

        return {
            "image"         : image,
            "input_ids"     : input_ids,
            "attention_mask": mask,
            "yesno"         : torch.tensor(yn_label, dtype=torch.long),
            "is_yesno"      : torch.tensor(yn_flag,  dtype=torch.bool),
            "answer"        : gen_label,
        }