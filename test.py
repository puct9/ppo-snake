import random
from pathlib import Path

import numpy as np
import torch

import snake
from rollout import rollout
from snake import Direction
from utils import load_from_dir

np.set_printoptions(suppress=True, precision=3)

LOAD_DIR = Path("save") / "Nov12_22-52-51_DESKTOP-O2TEBKS"


NET, NET_OPTIM, STATE_STATS, REWARD_STATS, CONFIG = load_from_dir(LOAD_DIR)


def test():
    dims = CONFIG.height, CONFIG.width
    _, _, _, _, adv, perf = rollout(
        NET, dims, STATE_STATS, REWARD_STATS, CONFIG.gamma, CONFIG.lam
    )
    print(perf)


@torch.no_grad()
def interactive():
    dims = CONFIG.height, CONFIG.width

    reward_sum = 0
    game = getattr(snake, CONFIG.game)(*dims)
    cont = True
    while cont:
        # print(game.state[:-4].reshape(2, *dims))
        print(game)
        state = game.state
        state = (state - STATE_STATS.mean) / np.sqrt(STATE_STATS.var)
        t = torch.from_numpy(state).unsqueeze(0)
        policy, value = NET.forward(t)
        policy = policy.exp().squeeze(0).numpy()
        print(policy)
        print(value.item())

        # Move
        input()
        move = random.choices(list(Direction), weights=policy, k=1)[0]
        cont, reward = game.move(move)
        reward_sum += reward
        print(reward)
    print(reward_sum)


interactive()
