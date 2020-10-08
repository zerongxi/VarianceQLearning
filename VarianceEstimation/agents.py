import numpy as np
from utils import init_arr_of_lists


class QLearning:

    def __init__(
            self,
            state_dims,
            n_actions,
            gamma,
            alpha,
            sucb=False,
            c=1.645,
            std_init=20.,
    ):
        self._n_actions = n_actions
        self._gamma = gamma
        self._alpha = alpha
        self._q = np.zeros((*state_dims, n_actions), np.float32)
        self._sucb = sucb
        if sucb:
            self._std = np.ones_like(self._q) * std_init
            self.c = c

    def update(self, state, state_, action, reward, episode):
        q_ = 0. if state_ is None else self._gamma * np.max(self._q[state_])
        q_ += reward
        idx = (*state, action)
        q = self._q[idx]
        self._q[idx] += self._alpha * (q_ - q)
        if self._sucb:
            e = self._q[idx]
            std = self._std[idx]
            self._std[idx] = self._calc_std(q, q_, e, std, 1 - self._alpha)
        return q_ - q

    def policy(self, state, training, eps):
        q, std = None, None
        if np.random.random() < eps:
            action = np.random.randint(0, self._n_actions)
        else:
            q = self._q[state]
            p = q
            if training and self._sucb:
                std = self._std[state]
                p = p + std * self.c
            action = np.argmax(p)
        return action, q, std

    @staticmethod
    def _calc_std(q, q_, e, std, beta):
        var_old = np.power(std, 2) + np.power(e - q, 2)
        var_new = np.power(q_ - e, 2)
        std_ = np.sqrt(beta * var_old + (1 - beta) * var_new)
        return std_
