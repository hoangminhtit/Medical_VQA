import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, T5Tokenizer
import torchvision.transforms as transforms


class MedicalVQADataset(Dataset):
    # OpenCLIP normalisation used by BioMedCLIP
    _MEAN = (0.48145466, 0.4578275,  0.40821073)
    _STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, data, image_folder=None):
        """
        Args:
            data        : list of dicts OR HuggingFace Dataset.
                          Expected keys: 'image', 'question', 'answer'.
                          'image' may be a filename (str) or a PIL Image.
            image_folder: base folder when 'image' is a filename.
                          Pass None (default) when using HuggingFace datasets
                          where 'image' is already a PIL Image.
        """
        self.data         = data
        self.image_folder = image_folder

        # Use BioMedCLIP tokenizer for questions (needed for encoder)
        self.question_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        
        # Use T5 tokenizer for answers (matches the decoder)
        self.answer_tokenizer = T5Tokenizer.from_pretrained("t5-base")

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

        # ── Image ───────────────────────────────────────────────────
        raw = sample["image"]
        if isinstance(raw, str):
            raw = Image.open(f"{self.image_folder}/{raw}")
        image = self.transform(raw.convert("RGB"))

        # ── Tokenise question ───────────────────────────────────────
        question = sample["question"]
        q_tokens = self.question_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        input_ids = q_tokens["input_ids"].squeeze(0)
        mask      = q_tokens["attention_mask"].squeeze(0)

        # ── Answer labels ───────────────────────────────────────────
        answer     = sample["answer"]
        yn_flag    = self.is_yesno(answer)
        yn_label   = 1 if (yn_flag and answer.strip().lower() == "yes") else 0

        # Use T5 tokenizer for answer labels to match T5 decoder vocabulary
        gen_tokens = self.answer_tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        gen_label = gen_tokens["input_ids"].squeeze(0)  # (16,)

        return {
            "image"         : image,
            "input_ids"     : input_ids,
            "attention_mask": mask,
            "yesno"         : torch.tensor(yn_label, dtype=torch.long),
            "is_yesno"      : torch.tensor(yn_flag,  dtype=torch.bool),
            "answer"        : gen_label,
        }