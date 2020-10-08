import numpy as np
import gym
import multiprocessing as mp
from matplotlib import pyplot
import h5py
import pandas as pd
import sys
import os
import pickle

from agents import QLearning
from utils import LinearScheduler


ENV_ID = "CartPole-v1"
CONFIG = dict(
    units=(4.8/12., 6./8., .42/16., 4./12.),
    lows=np.array((-2.4, -3., -.21, -2.)),
    highs=np.array((2.399, 2.99, .2099, 1.99)),
    gamma=1.,
    alpha=.1,
    n_episodes=30000,
    eval_step=200,
    eps_eval=0.05,
    cache_file="cartpole.h5"
)
N_CPUS = 4
N_TRIALS = 9
EVAL_TRIALS = 10


class Discrete:

    def __init__(self, units, lows, highs):
        self.units = units
        self.lows = lows
        self.highs = highs

    def obs2state(self, obs):
        obs = np.maximum(obs, self.lows)
        obs = np.minimum(obs, self.highs)
        obs = (obs - self.lows) / self.units
        return tuple(obs.astype(np.int))


def roll_out(agent, env, obs2state, training, eps, episode=None):
    terminal = False
    state = obs2state(env.reset())
    reward = 0.
    behavior = list()
    std_history = list()
    td_history = list()
    while not terminal:
        action, q, std = agent.policy(state, training, eps)
        obs, r, terminal, _ = env.step(action)
        state_ = obs2state(obs)
        if training:
            td = agent.update(state, None if terminal else state_, action, r, episode)
            if q is not None and std is not None:
                behavior.append(action != np.argmax(q))
                std_history.append(std[action])
                td_history.append(td)
        state = state_
        reward += r

    behavior = np.mean(behavior) if len(behavior) > 0 else None
    return reward, behavior, std_history, td_history


def q_learning(
        env_id,
        eps_init,
        eps_final,
        eps_steps,
        sucb,
        c=None,
        std_init=None,
):
    env = gym.make(env_id)
    config = CONFIG
    lows = config["lows"]
    highs = config["highs"]
    units = config["units"]
    gamma = config["gamma"]
    alpha = config["alpha"]
    n_episodes = config["n_episodes"]
    eval_step = config["eval_step"]
    eps_eval = config["eps_eval"]
    obs2state = Discrete(units, lows, highs).obs2state
    state_dims = np.array(obs2state(highs)) + 1
    kwargs = dict(
        state_dims=state_dims,
        n_actions=env.action_space.n,
        gamma=gamma,
        alpha=alpha,
        sucb=sucb,
        c=c,
        std_init=std_init,
    )
    agent = QLearning(**kwargs)
    reward, behavior = list(), list()
    behavior_temp = list()

    eps = LinearScheduler(eps_init, eps_final, eps_steps)

    for episode in range(n_episodes):
        _, b, std, td = roll_out(agent, env, obs2state, True, eps.get(), episode)
        if b is not None:
            behavior_temp.append(b)

        eps.step()
        if episode % eval_step == eval_step - 1:
            r = [roll_out(agent, env, obs2state, False, eps_eval)[0] for _ in range(EVAL_TRIALS)]
            reward.append(np.mean(r))
            if len(behavior_temp) > 0:
                behavior.append(np.mean(behavior_temp))
            else:
                behavior.append(np.nan)
            behavior_temp = list()
    reward = np.array(reward)
    if sucb:
        behavior = np.array(behavior, np.float32)
        while np.isnan(behavior).any():
            idx = np.argwhere(np.isnan(behavior))
            behavior[idx] = behavior[idx + 1]
    else:
        behavior = None
    return reward, behavior


def main(env_id):
    config = CONFIG
    cache_file = config["cache_file"]
    if not os.path.exists(cache_file):
        sets = dict()
        sets.update({
            "$\sigma$-warm_up": (env_id, 1., .0, 5000, True, .5, 0.),
            "$\sigma$-init_std": (env_id, 0., 0., 10, True, 1.5, 5000.),
            "$\sigma$-combined": (env_id, 1., 0., 5000, True, .5, 5000.),
        })
        sets.update({"$\epsilon$-greedy": (env_id, 1., 0.1, 5000, False)})

        labels, args = zip(*sets.items())
        reward = dict()
        behavior = dict()

        with mp.Pool(N_CPUS) as pool:
            r, b = zip(*pool.starmap(q_learning, [e for e in args for _ in range(N_TRIALS)]))
        for i, label in enumerate(labels):
            beg = i * N_TRIALS
            end = (i + 1) * N_TRIALS
            reward.update({label: np.stack(r[beg:end], axis=-1)})
            if b[beg] is not None:
                behavior.update({label: np.stack(b[beg:end], axis=-1)})

        with h5py.File(cache_file, "w") as h5f:
            for key, data in zip(("reward", "behavior"), (reward, behavior)):
                grp = h5f.create_group(key)
                for label, item in data.items():
                    grp[label] = item
    with h5py.File(cache_file, "r") as h5f:
        reward = dict([(k, np.array(v)) for k, v in h5f["reward"].items()])
        behavior = dict([(k, np.array(v)) for k, v in h5f["behavior"].items()])
    n_episodes = config["n_episodes"]
    eval_step = config["eval_step"]
    xticks = np.arange(eval_step, 1 + n_episodes, eval_step) / 1000.
    plot(reward, xticks, "reward").savefig("cartpole_reward.pdf")
    plot(behavior, xticks, "exploration rate").savefig("cartpole_behavior.pdf")
    print("Done!")


def plot(data, xticks, label):
    fig = pyplot.figure(figsize=(6, 4))
    keys = list(data.keys())
    percentiles = [np.percentile(data[k], [25, 50, 75], axis=1) for k in keys]
    colors = {
        "$\epsilon$-greedy": "gray",
        "$\sigma$-warm_up": "green",
        "$\sigma$-init_std": "red",
        "$\sigma$-combined": "purple",
    }
    ymax = 0.
    for key, p in zip(keys, percentiles):
        color = "tab:" + colors[key]
        low, median, high = [
            pd.DataFrame(u).rolling(window=20, min_periods=1).mean().to_numpy()[:, 0] for u in p
        ]
        """
        if np.max(median) < 450.:
            continue
        print(key, np.max(median), np.argmax(median) * 200)
        """
        pyplot.plot(xticks, median, label=key, color=color)
        pyplot.fill_between(xticks, low, high, facecolor=color, alpha=.4)
        pyplot.xlabel("episode (thousand)")
        pyplot.ylabel(label)
        ymax = 500. # max(ymax, np.max(high))
    pyplot.xlim(xmin=xticks[0], xmax=xticks[-1])
    pyplot.ylim(ymin=0, ymax=ymax)
    pyplot.legend()
    pyplot.tight_layout()
    return fig


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "CartPole-v1"
    main(env_id)

