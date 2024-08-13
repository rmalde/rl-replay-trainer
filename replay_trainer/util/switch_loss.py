import torch
import torch.nn.functional as F

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