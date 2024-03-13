import torch

from src.dataset import *

def test_dataset():
    dataset = Fruits(file = "utils/train_fruits.csv", path = "./")
    sample = dataset[0]
    assert sample.shape == (3, 100, 100)
    assert len(dataset) == 67692

    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0

    dataset = Fruits(file = "utils/test_fruits.csv", path = "./")
    sample = dataset[0]
    assert sample.shape == (3, 100, 100)
    assert len(dataset) == 22688

    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0