import numpy as np


class LinearScheduler:

    def __init__(self, init_val, final_val, steps):
        self.init_val = init_val
        self.final_val = final_val
        self.delta = (init_val - final_val) / steps
        self.val = init_val

    def get(self):
        return self.val

    def step(self, step=None):
        if step is None:
            self.val = max(self.val - self.delta, self.final_val)
        else:
            self.val = max(self.init_val - step * self.delta, self.final_val)
        return

    def reset(self):
        self.step(0)
        return


class SumTree:

    def __init__(self, capacity):
        self._depth = int(np.ceil(np.log2(capacity)))
        capacity = 2 ** self._depth
        self._capacity = capacity
        self._tree = np.zeros(capacity * 2 - 1, dtype=np.float32)
        return

    def update(self, nodes, values):
        idx = nodes + self._capacity - 1
        delta = values - self._tree[idx]

        to_update = [idx]
        for _ in range(self._depth):
            idx = (idx - 1) // 2
            to_update.append(idx)
        to_update = np.concatenate(to_update, axis=0)
        np.add.at(self._tree, to_update, np.tile(delta, self._depth + 1))

    def get_leaf(self, values):
        idx = np.zeros_like(values, dtype=np.int)
        for _ in range(self._depth):
            idx = idx * 2 + 1
            right = values > self._tree[idx]
            values[right] -= self._tree[idx[right]]
            idx[right] += 1
        nodes = idx - self._capacity + 1
        return nodes, np.copy(self._tree[idx])

    def sum(self):
        return self._tree[0]

    def max(self):
        return np.max(self._tree[-self._capacity:])

    def refresh(self):
        values = self._tree[self._capacity - 1:]
        update = list()
        while values.size > 1:
            values = np.reshape(values, (-1, 2))
            values = np.sum(values, axis=-1)
            update.append(values)
        update = list(reversed(update))
        update = np.concatenate(update, axis=0)
        self._tree[:self._capacity - 1] = update
