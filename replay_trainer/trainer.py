import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.rich import tqdm

from replay_trainer.util import (
    WandbLogger,
    CheckpointManager,
    ClassificationMetrics,
    RegressionMetrics,
)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        config: dict = None,
        device="cpu",
        objective="classification",
    ):
        if config is not None:
            self._load_config(config)

        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.scheduler_max_t)

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.wandb_logger = WandbLogger(self.wandb_project)
        self.wandb_logger.watch(self.model)
        self.objective = objective
        if objective == "classification":
            self.metrics = ClassificationMetrics(
                len(self.train_loader), len(self.test_loader)
            )
            self.criterion = F.cross_entropy
        elif objective == "regression":
            self.metrics = RegressionMetrics(
                len(self.train_loader), len(self.test_loader)
            )
            self.criterion = F.mse_loss
            # self.criterion = F.binary_cross_entropy
        else:
            raise ValueError(f"Invalid objective: {objective}")

    def _load_config(self, config: dict):
        self.learning_rate = config.get("learning_rate", 5e-3)
        self.num_epochs = config.get("num_epochs", 10)
        self.wandb_project = config.get("wandb_project", "rl-replay-trainer")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.scheduler_max_t = config.get("scheduler_max_t", 20)

    def train(self):

        for epoch in range(self.num_epochs):
            self.model.train()

            all_outputs = []
            all_targets = []

            # Training loop
            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch",
            ) as pbar:
                for batch in self.train_loader:
                    obs, target = batch
                    obs = obs.to(self.device)
                    target = target.to(self.device).squeeze()

                    self.optimizer.zero_grad()
                    # classification
                    # (b, obs_size) -> (b, action_size)
                    # regression
                    # (b, obs_size) -> (b, 1)
                    outputs = self.model(obs)

                    loss = self.criterion(outputs, target)
                    loss.backward()
                    if self.objective == "classification":
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                    self.metrics.update_train(loss.item(), outputs, target)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

                    all_outputs.append(outputs)
                    all_targets.append(target)

            if self.objective == "regression":
                self._save_plot(all_outputs, all_targets, "train_plot.png")
            # metrics
            self.evaluate()
            metrics_dict = self.metrics.to_dict()
            self.wandb_logger.log(metrics_dict)
            print(self.metrics)
            checkpoint_metric = (
                metrics_dict["Test Accuracy"]
                if self.objective == "classification"
                else 1-metrics_dict["Test Mean Absolute Error"]
            )
            self.checkpoint_manager.save_checkpoint(
                self.model, epoch, checkpoint_metric
            )
            self.metrics.reset()

    def _save_plot(self, all_outputs, all_targets, filename):
        all_outputs = torch.cat(all_outputs).detach().cpu().numpy()
        all_targets = torch.cat(all_targets).detach().cpu().numpy()
        # save scatterplot to scatterplot.png
        # Create a scatter plot
        plt.figure()
        sns.violinplot(x=all_targets, y=all_outputs)
        plt.title('Violin Plot')
        plt.xlabel('Targets')
        plt.ylabel('Outputs')

        # Save the plot to a PNG file
        plt.savefig(filename)

        print("Saved scatterplot to " , filename)

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                obs, target = batch
                obs = obs.to(self.device)
                target = target.to(self.device).squeeze()

                # Forward pass
                outputs = self.model(obs)

                loss = self.criterion(outputs, target)

                self.metrics.update_test(loss.item(), outputs, target)

                all_outputs.append(outputs)
                all_targets.append(target)


            if self.objective == "regression":
                self._save_plot(all_outputs, all_targets, "test_plot.png")
