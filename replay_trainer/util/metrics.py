import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Metrics(ABC):
    def __init__(self, len_train: int, len_test: int):
        self.len_train = len_train
        self.len_test = len_test

        self.reset()

    def reset(self):
        self.train_loss = 0
        self.test_loss = 0

    @abstractmethod
    def update_train(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor
    ):
        """Update metrics based on the current batch."""
        pass

    @abstractmethod
    def update_test(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor, **kwargs
    ):
        """Update metrics based on the current batch."""
        pass

    @abstractmethod
    def to_dict(self):
        """Return the computed metrics. For wandb logging."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the metrics."""
        pass


class ClassificationMetrics(Metrics):
    def __init__(self, len_train: int, len_test: int):
        super().__init__(len_train, len_test)

    def reset(self):
        super().reset()

        self.total_train = 0
        self.correct_top1_train = 0

        self.total_test = 0
        self.correct_top1_test = 0
        self.correct_top5_test = 0
        self.correct_switch_test = 0
        self.total_switch_test = 0

    def update_train(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor
    ):
        self.total_train += target.size(0)
        self.train_loss += loss
        # top 1
        _, predicted_top1 = torch.max(outputs.data, 1)
        self.correct_top1_train += (predicted_top1 == target).sum().item()

    def update_test(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor
    ):

        self.total_test += target.size(0)
        self.test_loss += loss
        # top 1
        _, predicted_top1 = torch.max(outputs.data, 1)
        self.correct_top1_test += (predicted_top1 == target).sum().item()
        # top 5
        _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
        self.correct_top5_test += (predicted_top5 == target.unsqueeze(1)).sum().item()
        # switch
        last_actions = actions[:, -1].squeeze()  # (batch_size, )
        self.total_switch_test += (last_actions != target).sum().item()
        self.correct_switch_test += (
            ((last_actions != target) & (predicted_top1 == target)).sum().item()
        )

    def to_dict(self) -> dict:
        return {
            "Train Loss": self.train_loss / self.len_train,
            "Test Loss": self.test_loss / self.len_test,
            "Train Accuracy": 100 * self.correct_top1_train / self.total_train,
            "Test Accuracy": 100 * self.correct_top1_test / self.total_test,
            "Test Top-5 Accuracy": 100 * self.correct_top5_test / self.total_test,
            "Test Switch Accuracy": 100
            * self.correct_switch_test
            / self.total_switch_test,
        }

    def __repr__(self):
        return_str = ""
        return_str += f"Train L: {(self.train_loss / self.len_train):.4f}, "
        return_str += f"Test L: {(self.test_loss / self.len_test):.4f}, "
        return_str += (
            f"Train Acc: {(100 * self.correct_top1_train / self.total_train):.2f}%, "
        )
        return_str += (
            f"Test Acc: {(100 * self.correct_top1_test / self.total_test):.2f}%, "
        )
        return_str += (
            f"Top-5 Acc: {(100 * self.correct_top5_test / self.total_test):.2f}%, "
        )
        return_str += f"Switch Acc: {(100 * self.correct_switch_test / self.total_switch_test):.2f}%"
        return return_str


class RegressionMetrics(Metrics):
    def __init__(self, len_train: int, len_test: int):
        super().__init__(len_train, len_test)

    def reset(self):
        super().reset()
        self.total_train = 0
        self.total_test = 0
        self.mae_train = 0
        self.mae_test = 0

    def update_train(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor
    ):
        self.total_train += target.size(0)
        self.train_loss += loss
        self.mae_train += torch.abs(target - outputs).sum().item()

    def update_test(
        self, loss: float, outputs: torch.Tensor, target: torch.Tensor
    ):
        self.total_test += target.size(0)
        self.test_loss += loss
        self.mae_test += torch.abs(target - outputs).sum().item()

    def to_dict(self) -> dict:
        return {
            "Train loss": self.train_loss / self.len_train,
            "Test loss": self.test_loss / self.len_test,
            "Train Mean Absolute Error" : self.mae_train / self.total_train,
            "Test Mean Absolute Error" : self.mae_test / self.total_test
        }

    def __repr__(self):
        return_str = ""
        return_str += f"Train L: {(self.train_loss / self.len_train):.4f}, "
        return_str += f"Test L: {(self.test_loss / self.len_test):.4f}, "
        return_str += f"Train Abs Error: {(self.mae_train / self.total_train):.4f}, "
        return_str += f"Test Abs Error: {(self.mae_test / self.total_test):.4f}"
        return return_str
