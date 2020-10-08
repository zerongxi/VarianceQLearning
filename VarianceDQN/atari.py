import gym
import numpy as np
import cv2


class Atari:

    def __init__(self, env_id):
        env_id += "NoFrameskip-v4"
        self.env = gym.make(env_id)
        self.skip = 4
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84),
            dtype=self.env.observation_space.dtype,
        )
        self.action_space = self.env.action_space

        self.lives = 0
        self.lost_live = False
        self.max_noop = 30
        self.state = np.zeros((4, *self.observation_space.shape), dtype=np.uint8)

        self.max_tick = 30 * 60 * 60 // 4
        self.tick = 0

        self.env.reset()
        self.max_lives = self.env.step(1)[-1]["ale.lives"]

    def reset(self):
        self.lost_live = False
        self.lives = self.max_lives
        self.tick = 0
        frames = [self.env.reset()]
        for _ in range(self.state.shape[0]):
            self._update_state(frames)
        n_noop = np.random.randint(self.max_noop)
        for _ in range(n_noop):
            self.step(0)
        if self.lives == self.max_lives:
            return self.state
        return self.reset()

    def step(self, action):
        frames = list()
        reward = 0.

        actions = [action] * 4
        if self.lost_live:
            actions = [1] + [action]
        for a in actions:
            f, r, terminal, info = self.env.step(a)
            frames.append(f)
            reward += r
            self.lost_live = info["ale.lives"] < self.lives
            if terminal or self.lost_live:
                break
        self._update_state(frames)

        if self.lost_live:
            self.lives = info["ale.lives"]
        self.tick += 1
        if self.tick > self.max_tick:
            terminal = True
        return self.state, reward, terminal, self.lost_live

    def _update_state(self, frames):
        frm = np.max(np.stack([
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
            for f in frames[-2:]
        ]), axis=0)
        self.state = np.append(self.state[1:], np.expand_dims(frm, 0), 0)

    def close(self):
        self.env.close()
