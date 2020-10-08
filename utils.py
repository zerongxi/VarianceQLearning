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


def init_arr_of_lists(shape):
    arr = np.empty(shape, dtype=np.object)
    arr = arr.flatten()
    for i, _ in enumerate(arr):
        arr[i] = list()
    arr = arr.reshape(shape)
    return arr
