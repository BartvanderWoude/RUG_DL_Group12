import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms


class Fruits(Dataset):
    def __init__(self, file, path="./"):
        self.df = pd.read_csv(path + file)
        self.path = path
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.df.iloc[idx, 0])
        image = read_image(img_path)
        image = self.transform(image)
        # label = self.img_labels.iloc[idx, 1]

        return image


if __name__ == "__main__":
    dataset = Fruits(file="utils/train_fruits.csv", path="../")
    for sample in dataset:
        print(sample.shape)
        print(sample)
        print(torch.max(sample))
        print(torch.min(sample))
        break
