import os
import torch


class Logger:
    """Pipeline for writing logs to files as well as saving models."""
    def __init__(self):
        self.logs_path = "output/logs/"
        self.model_path = "output/models/"

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.training = open(self.logs_path + "training_loss.csv", "w")
        self.training.write("epoch,loss\n")

        self.validation = open(self.logs_path + "validation_loss.csv", "w")
        self.validation.write("epoch,loss\n")

    def save_model(self, model: torch.nn.Module, fold, epoch):
        torch.save(model.state_dict(), "%smodel-f%s-e%s.pth" % (self.model_path,
                                                                str(fold),
                                                                str(epoch)))

    def log_training_loss(self, epoch, loss):
        self.training.write(str(epoch) + "," + str(loss) + "\n")

    def log_validation_loss(self, epoch, loss):
        self.validation.write(str(epoch) + "," + str(loss) + "\n")
