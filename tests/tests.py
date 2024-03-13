import torch

from src.dataset import Fruits


def test_traindataset():
    dataset = Fruits(file="utils/train_fruits.csv")
    sample = dataset[0]
    assert sample.shape == (3, 100, 100)
    assert len(dataset) == 67692

    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0


def test_testdataset():
    dataset = Fruits(file="utils/test_fruits.csv")
    sample = dataset[0]
    assert sample.shape == (3, 100, 100)
    assert len(dataset) == 22688

    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0