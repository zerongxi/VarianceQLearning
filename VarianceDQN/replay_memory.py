import torch
import numpy as np
from threading import Thread, Lock
from queue import Full, Empty, Queue
import time
from utils import SumTree


class UniformReplayMemory:

    def __init__(self, **kwargs):
        self.capacity = kwargs["capacity"]
        self.history_len = kwargs["history_len"]
        self.batch_sz = kwargs["batch_sz"]
        self.device = kwargs["device"]

        self.frame = np.zeros((self.capacity, *kwargs["state_shape"]), dtype=kwargs["state_dtype"])
        self.action = np.zeros(self.capacity, dtype=np.uint8)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminal = np.zeros(self.capacity, dtype=np.bool)

        self.ptr = 0
        self.full = False

    def put(self, frame, action, reward, terminal):
        if self.ptr == self.capacity:
            self.ptr = 0
            self.full = True
        self.frame[self.ptr] = frame
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.terminal[self.ptr] = terminal
        self.ptr += 1

    def sample(self):
        low = self.history_len - 1
        high = self.capacity - 1 if self.full else self.ptr - 1
        idx = np.random.randint(low, high, self.batch_sz)
        batch = (
            torch.from_numpy(np.stack(
                [self.frame[u:v] for u, v in zip(idx + 1 - self.history_len, idx + 1)]
            )).to(self.device).float(),
            torch.from_numpy(np.stack(
                [self.frame[u:v] for u, v in zip(idx + 2 - self.history_len, idx + 2)]
            )).to(self.device).float(),
            torch.from_numpy(self.action[idx]).to(self.device).long(),
            torch.from_numpy(self.reward[idx]).to(self.device),
            torch.from_numpy(self.terminal[idx]).to(self.device),
        )
        return batch


class PrioritizedReplayMemory(UniformReplayMemory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = np.array(kwargs["alpha"], dtype=np.float32)
        self.beta = kwargs["beta"]
        self.priority_eps = np.array(kwargs["priority_eps"], dtype=np.float32)
        self.priority_upper = np.array(kwargs["priority_upper"], dtype=np.float32)
        self.tree = SumTree(self.capacity)

    def update_priority(self, idx, priority):
        if priority is None:
            idx = np.array([idx])
            priority = np.array([self.priority_upper])
        else:
            priority = priority + self.priority_eps
            priority = np.minimum(priority, self.priority_upper)
        priority = np.power(priority, self.alpha)
        self.tree.update(idx, priority)
        return

    def put(self, frame, action, reward, terminal):
        if self.ptr == self.capacity:
            self.tree.refresh()
            self.ptr = 0
            self.full = True
        self.frame[self.ptr] = frame
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.terminal[self.ptr] = terminal
        self.update_priority(self.ptr, None)
        self.ptr += 1

    def sample(self):
        tree_sum = self.tree.sum()
        seg_range = tree_sum / self.batch_sz
        seg_base = np.arange(self.batch_sz, dtype=np.float32) * seg_range
        values = np.random.random(self.batch_sz) * seg_range + seg_base
        idx, priority = self.tree.get_leaf(values)
        low = self.history_len - 1
        high = self.capacity - 1 if self.full else self.ptr - 1
        valid = np.logical_and(idx >= low, idx < high)
        idx = idx[valid]
        priority = priority[valid]
        idx, unique = np.unique(idx, return_index=True)
        priority = priority[unique]
        is_weight = np.power(
            tree_sum / np.array(self.capacity, dtype=np.float32) / (priority + 1e-6),
            np.array(self.beta.get(), dtype=np.float32)
        )
        is_weight /= np.max(is_weight)
        batch = (
            torch.from_numpy(np.stack(
                [self.frame[u:v] for u, v in zip(idx + 1 - self.history_len, idx + 1)]
            )).to(self.device).float(),
            torch.from_numpy(np.stack(
                [self.frame[u:v] for u, v in zip(idx + 2 - self.history_len, idx + 2)]
            )).to(self.device).float(),
            torch.from_numpy(self.action[idx]).to(self.device).long(),
            torch.from_numpy(self.reward[idx]).to(self.device),
            torch.from_numpy(self.terminal[idx]).to(self.device),
            torch.from_numpy(is_weight).to(self.device),
            idx,
        )
        self.beta.step()
        return batch
