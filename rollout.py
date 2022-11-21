import random
from typing import Sequence

import numpy as np
import torch

import snake
from neural import ActorCritic
from running_mean_std import RunningMeanStd
from snake import Direction
from utils import Config

State = np.ndarray
Action = int
LogProb = float
DiscountedReward = float
Advantage = float
RewardSum = float

DIRECTIONS = list(Direction)


@torch.no_grad()
def rollout(
    net: ActorCritic,
    state_stats: RunningMeanStd,
    reward_stats: RunningMeanStd,
    config: Config,
    *,
    max_steps: int = -1,
) -> tuple[
    Sequence[State],
    Sequence[Action],
    Sequence[LogProb],
    Sequence[DiscountedReward],
    Sequence[Advantage],
    RewardSum,
]:
    dims = config.height, config.width
    gamma = config.gamma
    lam = config.lam
    game_type = getattr(snake, config.game)

    states: list[State] = []
    actions: list[int] = []
    action_log_probs: list[float] = []
    value_preds: list[float] = []
    rewards: list[float] = []

    game = game_type(*dims)
    cont = True

    step = 0
    state_stats_std = np.sqrt(state_stats.var)
    while step != max_steps and cont:
        # Get and normalize state
        state = game.state
        state = (state - state_stats.mean) / state_stats_std
        states.append(state)
        assert state.dtype == np.float32

        state_t = torch.from_numpy(state).unsqueeze(0)
        log_probs, value_pred = net(state_t)
        value_preds.append(value_pred.item())

        # Sample move from policy
        log_probs.squeeze_(0)
        probs = torch.exp(log_probs).numpy()
        action = random.choices([0, 1, 2, 3], weights=probs, k=1)[0]
        actions.append(action)
        action_log_probs.append(log_probs[action].item())

        cont, reward = game.move(DIRECTIONS[action])
        rewards.append(reward)

    # Pad the trajectory for correct advantage computation
    # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/ppo.py#L42
    if cont:
        state = game.state
        state = (state - state_stats.mean) / state_stats_std
        state_t = torch.from_numpy(state).unsqueeze(0)
        _, value_pred = net(state_t)
        last_val = value_pred.item()
        rewards.append(last_val)
        value_preds.append(last_val)
    else:
        rewards.append(0)
        value_preds.append(0)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    action_log_probs = np.array(action_log_probs, dtype=np.float32)
    value_preds = np.array(value_preds, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)

    unnormalised_rewards_sum = rewards.sum()  # For tracking progress

    # Update running state stats
    state_stats.update(states)

    # Compute discounted rewards for normalisation
    discounted_rewards = discount(rewards, gamma)[:-1]
    # Update reward_stats using `discounted_rewards`
    reward_stats.update(discounted_rewards)
    # Normalise discounted rewards without subtracting mean
    # Also clip to range of [-10, 10]
    rewards = np.clip(
        rewards / np.sqrt(reward_stats.var),
        -10,
        10,
    )

    # Compute GAE
    deltas = -value_preds[:-1] + rewards[:-1] + gamma * value_preds[1:]
    advantages = discount(deltas, lam * gamma)

    # Recompute discounted rewards for use as value function targets
    # I still don't know why this is calculated like this
    # discounted_rewards = value_preds[:-1] + advantages
    # Alternatively
    discounted_rewards = discount(rewards, gamma)[:-1]

    return (
        states,
        actions,
        action_log_probs,
        discounted_rewards,
        advantages,
        unnormalised_rewards_sum,
    )


def discount(seq: np.ndarray, discount_factor: float) -> None:
    res = seq.copy()
    if len(res) == 1:
        return res
    for i in range(len(res) - 2, -1, -1):
        res[i] += discount_factor * res[i + 1]
    return res
