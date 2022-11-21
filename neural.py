import numpy as np
import torch
import torch.nn as nn

PolicyOut = torch.Tensor
ValueOut = torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, *dims: int, moves: int, fc_size: int = 32) -> None:
        super().__init__()
        self.common = nn.Sequential(
            # Flatten [ N 1 H W ] -> [ N H*W ]
            nn.Flatten(),
            # nn.Linear(np.product(dims), fc_size),
            # nn.ReLU(),
            # nn.Linear(fc_size, fc_size),
            # nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(np.product(dims), fc_size),
            nn.Tanh(),
            # nn.Linear(fc_size, fc_size),
            # nn.Tanh(),
            nn.Linear(fc_size, fc_size),
            nn.Tanh(),
            nn.Linear(fc_size, moves),
            nn.LogSoftmax(1),
        )
        self.critic = nn.Sequential(
            nn.Linear(np.product(dims), fc_size),
            nn.Tanh(),
            # nn.Linear(fc_size, fc_size),
            # nn.Tanh(),
            nn.Linear(fc_size, fc_size),
            nn.Tanh(),
            nn.Linear(fc_size, 1),
        )

    def __call__(self, *args, **kwargs) -> tuple[PolicyOut, ValueOut]:
        return super().__call__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple[PolicyOut, ValueOut]:
        x = self.common(x)
        a = self.actor(x)
        c = self.critic(x)
        return a, c


class ActorCriticCNN(nn.Module):
    def __init__(self, *dims: int, moves: int, fc_size: int = 32) -> None:
        super().__init__()
        # Dims should be (C, H, W)
        if len(dims) != 3:
            raise ValueError(f"Dims should be in form (C, H, W), not {dims}")
        self.common = nn.Sequential(
            nn.Conv2d(dims[0], 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(np.product(dims[1:]), fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, moves),
            nn.LogSoftmax(1),
        )
        self.critic = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(np.product(dims[1:]), fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[PolicyOut, ValueOut]:
        x = self.common(x)
        a = self.actor(x)
        c = self.critic(x)
        return a, c
