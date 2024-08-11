import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.rich import tqdm

from replay_trainer.util import WandbLogger, CheckpointManager


def custom_loss(outputs, target, last_actions):
    # outputs: (batch_size, num_actions)
    # target: (batch_size)
    # last_actions: (batch_size, 1)
    ce_loss = F.cross_entropy(outputs, target)

    # penalize repeated previous action when you are wrong
    probs = F.softmax(outputs, dim=1)
    repeated_prob = probs.gather(1, last_actions).squeeze()
    changed_mask = (target != last_actions.squeeze()).float()
    wrong_repetition = (repeated_prob * changed_mask).sum()
    # penalty_strength = 200.0
    penalty_strength = 0.0
    penalty_term = penalty_strength * wrong_repetition / outputs.shape[0]

    return ce_loss + penalty_term



class Trainer:
    def __init__(
        self, model, train_loader, test_loader, config: dict = None, device="cpu"
    ):
        if config is not None:
            self._load_config(config)

        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        self.wandb_logger = WandbLogger(self.wandb_project)
        self.wandb_logger.watch(self.model)

    def _load_config(self, config: dict):
        self.learning_rate = config.get("learning_rate", 5e-3)
        self.num_epochs = config.get("num_epochs", 10)
        self.wandb_project = config.get("wandb_project", "rl-replay-trainer")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

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
                    self.optimizer.step()
                    train_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            # Average training loss
            train_loss /= len(self.train_loader)

            # Train accuracy
            accuracy = 100 * correct / total

            # Validation
            test_loss, test_accuracy, top_5_accuracy, switch_accuracy = self.evaluate()

            self.wandb_logger.log(
                {
                    "Train Loss": train_loss,
                    "Test Loss": test_loss,
                    "Train Accuracy": accuracy,
                    "Test Accuracy": test_accuracy,
                    "Test Top-5 Accuracy": top_5_accuracy,
                    "Test Switch Accuracy": switch_accuracy,
                }
            )
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train L: {train_loss:.4f}"
                + f", Test L: {test_loss:.4f}"
                + f", Train Acc: {accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Top-5 Acc: {top_5_accuracy:.2f}%"
                + f", Switch Acc: {switch_accuracy:.2f}%"
            )
            self.checkpoint_manager.save_checkpoint(self.model, epoch, accuracy)

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        correct_switch = 0
        total = 0
        total_switch = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                input_seq, target = batch
                actions = input_seq["actions"].to(self.device)
                obs = input_seq["obs"].to(self.device)
                target = target.to(self.device).squeeze()

                # Forward pass
                outputs = self.model(actions, obs)
                # outputs = outputs[:, -1, :]

                # Compute loss
                loss = self.criterion(outputs, target)
                test_loss += loss.item()

                # Calculate top-1 accuracy
                _, predicted_top1 = torch.max(outputs.data, 1)
                total += target.size(0)
                correct_top1 += (predicted_top1 == target).sum().item()

                # Calculate top-5 accuracy
                _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
                correct_top5 += (predicted_top5 == target.view(-1, 1)).sum().item()

                # Calculate switch accuracy
                # accuracy only when last action is different from target
                _, predicted_switch = torch.max(outputs.data, 1)
                last_actions = actions[:, -1].squeeze() # (batch_size, )
                total_switch += (last_actions != target).sum().item()
                correct_switch += ( (last_actions != target) & (predicted_switch == target) ).sum().item()


        # Average test loss and accuracies
        test_loss /= len(self.test_loader)
        accuracy_top1 = 100 * correct_top1 / total
        accuracy_top5 = 100 * correct_top5 / total
        accuracy_switch = 100 * correct_switch / total_switch

        return test_loss, accuracy_top1, accuracy_top5, accuracy_switch
