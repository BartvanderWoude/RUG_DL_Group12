import torch
import os

from src.dataset import Fruits
from src.logger import Logger


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


def test_logger():
    logger = Logger()
    logger.log_training_loss(2, 1, 0.5)
    logger.log_validation_loss(2, 1, 0.5)
    logger.save_model(torch.nn.Module(), 6, 1)
    logger.close()

    with open(logger.logs_path + "training_loss.csv", "r") as f:
        assert f.read() == "fold,epoch,loss\n2,1,0.5\n"
    with open(logger.logs_path + "validation_loss.csv", "r") as f:
        assert f.read() == "fold,epoch,loss\n2,1,0.5\n"
    assert os.path.exists(logger.models_path + "model-f6-e1.pth")
