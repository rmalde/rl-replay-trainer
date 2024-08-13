import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20)

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        self.wandb_logger = WandbLogger(self.wandb_project)
        self.wandb_logger.watch(self.model)

        if objective == "classification":
            self.metrics = ClassificationMetrics(len(self.train_loader), len(self.test_loader))
            self.criterion = F.cross_entropy
        elif objective == "regression":
            self.metrics = RegressionMetrics(len(self.train_loader), len(self.test_loader))
            self.criterion = F.mse_loss
        else:
            raise ValueError(f"Invalid objective: {objective}")

    def _load_config(self, config: dict):
        self.learning_rate = config.get("learning_rate", 5e-3)
        self.num_epochs = config.get("num_epochs", 10)
        self.wandb_project = config.get("wandb_project", "rl-replay-trainer")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

    def train(self):

        for epoch in range(self.num_epochs):
            self.model.train()

            # Training loop
            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch",
            ) as pbar:
                for batch in self.train_loader:
                    input_seq, target = batch
                    actions = input_seq["actions"].to(self.device)
                    obs = input_seq["obs"].to(self.device)
                    target = target.to(self.device).squeeze()

                    self.optimizer.zero_grad()

                    # (b, seq_len, num_actions)
                    outputs = self.model(actions, obs)
                    # outputs = outputs[:, -1, :]

                    loss = self.criterion(outputs, target)
                    loss.backward()

                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                    self.metrics.update_train(loss.item(), outputs, target)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            # metrics
            self.evaluate()
            metrics_dict = self.metrics.to_dict()
            self.wandb_logger.log(metrics_dict)
            print(self.metrics)
            self.checkpoint_manager.save_checkpoint(
                self.model, epoch, metrics_dict["Test Accuracy"]
            )
            self.metrics.reset()

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_seq, target = batch
                actions = input_seq["actions"].to(self.device)
                obs = input_seq["obs"].to(self.device)
                target = target.to(self.device).squeeze()

                # Forward pass
                outputs = self.model(actions, obs)

                loss = self.criterion(outputs, target)

                self.metrics.update_test(loss.item(), outputs, target, actions=actions)
