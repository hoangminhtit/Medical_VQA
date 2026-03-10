import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import DatasetConfig

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

hf_dataset = load_dataset(DatasetConfig.PATH_VQA)

class VQA_Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]['image']
        question = self.data[index]['question']
        answer = self.data[index]['answer']

        if self.transform:
          image = self.transform(image)
        return image, question, answer


val_key = "validation" if "validation" in hf_dataset else "val"

train_data = hf_dataset['train']
test_data  = hf_dataset['test']
val_data   = hf_dataset[val_key]

train_dataset = VQA_Dataset(train_data, transform)
test_dataset  = VQA_Dataset(test_data,  transform)
val_dataset   = VQA_Dataset(val_data,   transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

if __name__=="__main__":
  for idx in range(0, 3):
    image, question, answer = train_dataset[idx]
    
    image = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Question: {question} \nAnswer: {answer}", fontsize=12)
    plt.axis('off')
    plt.show()