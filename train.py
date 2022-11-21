from __future__ import annotations

import socket
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam as Opt
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from neural import ActorCriticCNN as Model
from rollout import rollout
from running_mean_std import RunningMeanStd
from snake import SnakeGameCNN as Game
from utils import Config, MovingAverage, load_from_dir, save_to_dir


def main() -> None:
    name = (
        f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
        if len(sys.argv) == 1
        else sys.argv[1]
    )
    # Reduce `name` to the last part of the path if possible.
    # e.g., abc\\def -> def, abc/def -> def
    # This allows you to use autocomplete easily in the terminal
    name = Path(name).name

    save_path = Path(f"save/{name}")
    save_path.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) == 1:
        config = Config(
            lr=1e-4,
            height=8,
            width=8,
            gamma=0.99,
            lam=0.95,
            net_type=Model.__name__,
            optimizer=Opt.__name__,
            game=Game.__name__,
            ppo_update_epochs=4,
            minibatch_size=69420,
            min_samples=4096,
            fc_size=64,
        )
        dims = config.height, config.width

        state_shape = Game(*dims).state.shape
        net = Model(*state_shape, moves=4, fc_size=config.fc_size)
        net_optim = Opt(net.parameters(), lr=config.lr)

        state_stats = RunningMeanStd(shape=state_shape)
        reward_stats = RunningMeanStd(shape=())  # Scalar

    else:
        print(f"Loading {name}")
        net, net_optim, state_stats, reward_stats, config = load_from_dir(
            save_path
        )
        dims = config.height, config.width
        state_shape = state_stats.mean.shape

    # Logging
    writer = SummaryWriter(f"logs/{config.height}x{config.width}/{name}")
    if len(sys.argv) == 1:
        writer.add_graph(net, torch.rand(1, *state_shape))

    try:
        train(
            net,
            state_stats,
            reward_stats,
            net_optim,
            config,
            writer,
            save_path=save_path,
        )
    except KeyboardInterrupt:
        pass

    # Evaluate performance
    net.eval()
    avg = 0
    sample_size = 100
    for _ in range(sample_size):
        _, _, _, _, _, performance = rollout(
            net,
            state_stats,
            reward_stats,
            config,
        )
        avg += performance
    avg /= sample_size

    writer.add_hparams(
        asdict(config),
        {
            "hparam/perf": avg,
        },
    )

    # Save NN, optimiser state, running statistics, config
    save_to_dir(
        save_path,
        net,
        net_optim,
        state_stats,
        reward_stats,
        config,
    )


def train(
    net: Model,
    state_stats: RunningMeanStd,
    reward_stats: RunningMeanStd,
    net_optim: Optimizer,
    config: Config,
    writer: SummaryWriter,
    *,
    max_rollouts: int = -1,
    max_steps_per_rollout: int = -1,
    save_path: Path | None = None,
) -> None:
    save_path = save_path or Path()

    dims = config.height, config.width
    min_samples = config.min_samples
    ppo_update_epochs = config.ppo_update_epochs
    minibatch_size = config.minibatch_size

    score_avg = MovingAverage(100)

    state_shape = Game(*dims).state.shape
    mse_loss = nn.MSELoss()
    # Batch
    batch_states = np.zeros(shape=(0, *state_shape), dtype=np.float32)
    batch_actions = np.zeros(shape=(0,), dtype=np.int64)
    batch_old_log_probs = np.zeros(shape=(0,), dtype=np.float32)
    batch_discounted_rewards = np.zeros(shape=(0,), dtype=np.float32)
    batch_advantages = np.zeros(shape=(0,), dtype=np.float32)

    while config.step != max_rollouts:
        # Collect selfplay data
        net.eval()
        (
            rollout_states,
            rollout_actions,
            rollout_old_log_probs,
            rollout_discounted_rewards,
            rollout_advantages,
            performance,
        ) = rollout(
            net,
            state_stats,
            reward_stats,
            config,
            max_steps=max_steps_per_rollout,
        )
        batch_states = np.concatenate((batch_states, rollout_states), axis=0)
        batch_actions = np.concatenate(
            (batch_actions, rollout_actions), axis=0
        )
        batch_old_log_probs = np.concatenate(
            (batch_old_log_probs, rollout_old_log_probs), axis=0
        )
        batch_discounted_rewards = np.concatenate(
            (batch_discounted_rewards, rollout_discounted_rewards), axis=0
        )
        batch_advantages = np.concatenate(
            (batch_advantages, rollout_advantages), axis=0
        )

        # Update NN
        if batch_states.shape[0] < min_samples:
            continue

        config.step += 1
        net.train()

        states = torch.from_numpy(batch_states)
        actions = torch.from_numpy(batch_actions).unsqueeze(-1)
        old_log_probs = torch.from_numpy(batch_old_log_probs).unsqueeze(-1)
        discounted_rewards = torch.from_numpy(
            batch_discounted_rewards
        ).unsqueeze(-1)
        advantages_unnorm = torch.from_numpy(batch_advantages).unsqueeze(-1)

        assert states.shape[1:] == state_shape
        assert actions.shape[1:] == (1,)
        assert old_log_probs.shape[1:] == (1,)
        assert discounted_rewards.shape[1:] == (1,)
        assert advantages_unnorm.shape[1:] == (1,)

        # Normalize all advantages. Also possible to do this on the minibatch
        # level, but study shows it does not cause perf diff.
        with torch.no_grad():
            # Advantage norm
            advantages = (
                advantages_unnorm - advantages_unnorm.mean()
            ) / advantages_unnorm.std()

        for _ in range(ppo_update_epochs):
            # Shuffle data
            perm = torch.randperm(states.shape[0])

            for minibatch_idxs in torch.split(perm, minibatch_size):
                mb_states = states[minibatch_idxs]
                mb_actions = actions[minibatch_idxs]
                mb_old_log_probs = old_log_probs[minibatch_idxs]
                mb_discounted_rewards = discounted_rewards[minibatch_idxs]
                mb_advantages = advantages[minibatch_idxs]

                policy, value = net(mb_states)
                ppo_r = torch.exp(
                    policy.gather(1, mb_actions) - mb_old_log_probs
                )
                l_clip = torch.minimum(
                    # L^CPI
                    ppo_r * mb_advantages,
                    # clip(r, 1-eps, 1+eps) * A
                    torch.clip(
                        ppo_r,
                        1 - 0.1,
                        1 + 0.1,
                    )
                    * mb_advantages,
                )
                gain_policy = torch.mean(l_clip)

                # Entropy bonus for policy
                ppo_c2 = 0.01
                gain_entropy = -ppo_c2 * torch.mean(
                    torch.sum(torch.exp(policy) * policy, dim=1)
                )

                # Fit the value function
                loss_value = mse_loss(value, mb_discounted_rewards)

                net_optim.zero_grad()
                (-gain_policy - gain_entropy + loss_value).backward()
                net_optim.step()

        # Reset batches
        batch_states = np.zeros(shape=(0, *state_shape), dtype=np.float32)
        batch_actions = np.zeros(shape=(0,), dtype=np.int64)
        batch_old_log_probs = np.zeros(shape=(0,), dtype=np.float32)
        batch_discounted_rewards = np.zeros(shape=(0,), dtype=np.float32)
        batch_advantages = np.zeros(shape=(0,), dtype=np.float32)

        # Record metrics
        if config.step % 10 != 0:  # This stuff takes up lots of disk space
            continue
        with torch.no_grad():
            writer.add_histogram(
                "Advantages (raw)", advantages_unnorm, config.step
            )
            writer.add_histogram(
                "Advantages (normalised)", advantages, config.step
            )

            # Write rollout stats
            writer.add_histogram(
                "Value targets", discounted_rewards, config.step
            )

            # Write NN training stats
            policy, value = net(states)
            ppo_r = torch.exp(policy.gather(1, actions) - old_log_probs)
            entropy = -torch.sum(torch.exp(policy) * policy, dim=1)
            # Write histograms
            writer.add_histogram("Entropy", entropy, config.step)
            # Clip ratios otherwise histogram tensorboard histogram is bad
            writer.add_histogram(
                "Ratio (-A)", ppo_r[advantages < 0].clip(0.9, 1.1), config.step
            )
            writer.add_histogram(
                "Ratio (+A)", ppo_r[advantages > 0].clip(0.9, 1.1), config.step
            )
            # Condensed information as scalars
            writer.add_scalar(
                "Info/Entropy", entropy.mean().item(), config.step
            )
            writer.add_scalar(
                "Info/RatioNEG",
                ppo_r[advantages < 0].mean().item(),
                config.step,
            )
            writer.add_scalar(
                "Info/RatioPOS",
                ppo_r[advantages > 0].mean().item(),
                config.step,
            )
            # Value loss
            value_loss = mse_loss(value, discounted_rewards)
            writer.add_scalar("Loss/Value", value_loss.item(), config.step)

            writer.add_scalar("Episodic reward", performance, config.step)
            score_avg.update(performance)
            print(f"Step {config.step} average {score_avg.value:.3f}")

        # Save
        save_to_dir(
            save_path,
            net,
            net_optim,
            state_stats,
            reward_stats,
            config,
        )


if __name__ == "__main__":
    main()
