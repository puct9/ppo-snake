import json
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim
from torch.optim import Optimizer

import snake
import neural
from neural import ActorCritic
from running_mean_std import RunningMeanStd
from snake import SnakeGame


class MovingAverage:
    def __init__(self, size: int) -> None:
        self.nums = deque(maxlen=size)
        self.s = 0

    @property
    def value(self) -> float:
        return self.s / len(self.nums)

    def update(self, v: float) -> None:
        if len(self.nums) == self.nums.maxlen:
            self.s -= self.nums[0]
        self.nums.append(v)
        self.s += v


@dataclass
class Config:
    lr: float
    height: int
    width: int
    gamma: float
    lam: float
    net_type: str
    optimizer: str
    game: str
    ppo_update_epochs: int
    minibatch_size: int
    min_samples: int
    fc_size: int
    step: int = 0


def load_from_dir(
    path: Path,
) -> tuple[ActorCritic, Optimizer, RunningMeanStd, RunningMeanStd, Config]:
    config = json.load((path / "config.json").open())
    config = Config(
        **{k: config[k] for k in Config.__dataclass_fields__.keys()}
    )
    # Config like:
    # {
    #     "lr": 0.0001,
    #     "height": 6,
    #     "width": 6,
    #     "dims": "(6, 6)",
    #     "gamma": 0.99,
    #     "lam": 0.95,
    #     "net_type": "ActorCritic",
    #     "optimizer": "Adam",
    #     "game": "SnakeGame"
    #     "ppo_update_epochs": 3,
    #     "minibatch_size": 69420,
    #     "fc_size": 64
    # }
    dims = config.height, config.width

    # Net and optim
    net_type: type[ActorCritic] = getattr(neural, config.net_type)
    game_type: type[SnakeGame] = getattr(snake, config.game)
    net = net_type(
        *game_type(*dims).state.shape, moves=4, fc_size=config.fc_size
    )
    net.load_state_dict(torch.load(path / "save.pt"))
    net_optim: Optimizer = getattr(torch.optim, config.optimizer)(
        net.parameters()
    )
    net_optim.load_state_dict(torch.load(path / "save_optim.pt"))

    # Running stats
    state_stats = pickle.load((path / "state_stats.rms").open("rb"))
    reward_stats = pickle.load((path / "reward_stats.rms").open("rb"))

    return (
        net,
        net_optim,
        state_stats,
        reward_stats,
        config,
    )


def save_to_dir(
    path: Path,
    net: ActorCritic,
    net_optim: Optimizer,
    state_stats: RunningMeanStd,
    reward_stats: RunningMeanStd,
    config: Config,
) -> None:
    torch.save(net.state_dict(), path / "save.pt")
    torch.save(net_optim.state_dict(), path / "save_optim.pt")
    pickle.dump(state_stats, (path / "state_stats.rms").open("wb"))
    pickle.dump(reward_stats, (path / "reward_stats.rms").open("wb"))
    json.dump(asdict(config), (path / "config.json").open("w"))
