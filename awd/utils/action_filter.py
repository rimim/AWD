import torch


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency, num_envs, num_actions, device):
        self.last_actions = torch.zeros(
            num_envs,
            num_actions,
            dtype=torch.float,
            device=device,
            requires_grad=False,
        )
        self.current_actions = torch.zeros(
            num_envs,
            num_actions,
            dtype=torch.float,
            device=device,
            requires_grad=False,
        )
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
        )

    def push(self, actions):
        self.current_actions = actions

    def get_filtered_action(self):
        self.last_actions = (
            self.alpha * self.last_actions + (1 - self.alpha) * self.current_actions
        )
        return self.last_actions
