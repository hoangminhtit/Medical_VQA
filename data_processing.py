import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import get_args, DatasetConfig


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class VQA_Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]["image"]
        question = self.data[index]["question"]
        answer = self.data[index]["answer"]

        if self.transform:
            image = self.transform(image)
        return image, question, answer


def resolve_splits(hf_dataset):
    if "train" not in hf_dataset:
        raise ValueError(
            f"Dataset has no 'train' split. Available: {list(hf_dataset.keys())}"
        )

    train_split = hf_dataset["train"]

    if "validation" in hf_dataset:
        val_split = hf_dataset["validation"]
    elif "val" in hf_dataset:
        val_split = hf_dataset["val"]
    elif "test" in hf_dataset:
        val_split = hf_dataset["test"]
    else:
        split_ds = train_split.train_test_split(test_size=0.1, seed=42)
        train_split = split_ds["train"]
        val_split = split_ds["test"]

    if "test" in hf_dataset:
        test_split = hf_dataset["test"]
    elif "validation" in hf_dataset:
        test_split = hf_dataset["validation"]
    elif "val" in hf_dataset:
        test_split = hf_dataset["val"]
    else:
        split_ds = train_split.train_test_split(test_size=0.1, seed=123)
        train_split = split_ds["train"]
        test_split = split_ds["test"]

    return train_split, val_split, test_split


def build_dataloaders(batch_size: int, dataset_id: str = None):
    dataset_id = dataset_id or DatasetConfig.PATH_VQA
    hf_dataset = load_dataset(dataset_id)
    train_data, val_data, test_data = resolve_splits(hf_dataset)

    train_dataset = VQA_Dataset(train_data, transform)
    test_dataset = VQA_Dataset(test_data, transform)
    val_dataset = VQA_Dataset(val_data, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def main():
    config = get_args()
    train_dataset, _, _, _, _, _ = build_dataloaders(
        batch_size=config.batch_size,
        dataset_id=config.dataset
    )

    for idx in range(0, min(3, len(train_dataset))):
        image, question, answer = train_dataset[idx]

        image = image.permute(1, 2, 0).numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"Question: {question} \nAnswer: {answer}", fontsize=12)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
