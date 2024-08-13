import os
import torch

class CheckpointManager:
    def __init__(self, directory, max_checkpoints=5):
        self.directory = directory
        self.max_checkpoints = max_checkpoints
        self.best_accuracy = 0.0
        self.best_epoch = None
        self.checkpoints = []  # List to keep track of checkpoints

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
    
    def _best_epoch_path(self, epoch):
        best_epoch_string = f"best_model_epoch_{epoch + 1}.pt"
        return os.path.join(self.directory, best_epoch_string)

    def save_checkpoint(self, model, epoch, accuracy):
        checkpoint_path = os.path.join(self.directory, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        self.checkpoints.append((checkpoint_path, accuracy))

        # Maintain only the last 5 checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint, _ = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)

        # Check and save the best accuracy checkpoint
        if accuracy > self.best_accuracy:
            if self.best_epoch is not None:
                os.remove(self._best_epoch_path(self.best_epoch))

            self.best_accuracy = accuracy
            self.best_epoch = epoch
            
            best_checkpoint_path = self._best_epoch_path(epoch)
            print(f"New best test accuracy, saving {best_checkpoint_path}")
            torch.save(model.state_dict(), best_checkpoint_path)