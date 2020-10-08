import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from atari import Atari
from agent import DQNAgent
from replay_memory import UniformReplayMemory, PrioritizedReplayMemory
from utils import LinearScheduler
from threading import Thread, Lock
from queue import Queue, Full, Empty
import os


TENSORBOARD_FOLDER = "./log/"
CSV_FOLDER = "./result/"
MODEL_FOLDER = "./model/"


class Trainer:

    def __init__(self, kwargs):
        kwargs["env_cls"] = Atari
        env = kwargs["env_cls"](kwargs["env_id"])
        kwargs["state_shape"] = env.observation_space.shape
        kwargs["state_dtype"] = np.uint8
        kwargs["n_actions"] = env.action_space.n
        kwargs["device"] = torch.device(kwargs["device_id"])
        env.close()
        self.__dict__.update(kwargs)
        self.agent = DQNAgent(**kwargs)
        self.writer = SummaryWriter("./log/")
        self.cuda_eval = torch.cuda.Stream(self.device)

        mem_kwargs = dict(
            capacity=self.mem_capacity,
            history_len=self.history_len,
            state_shape=self.state_shape,
            state_dtype=self.state_dtype,
            batch_sz=self.batch_sz,
            alpha=self.mem_alpha,
            beta=LinearScheduler(self.mem_beta, 1., self.train_steps),
            priority_eps=self.mem_priority_eps,
            priority_upper=self.mem_priority_upper,
            prioritized_replay=self.prioritized_replay,
            device=self.device,
        )
        mem_cls = PrioritizedReplayMemory if self.prioritized_replay else UniformReplayMemory
        self.mem = mem_cls(**mem_kwargs)
        self.mem_lock = Lock()
        self.sync = Queue(maxsize=1)
        self.sync.put(None)

    def play_thread(self):
        env = self.env_cls(self.env_id)
        terminal = True
        eps = LinearScheduler(self.eps_init, self.eps_final, self.eps_steps)
        behavior = list()
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            for global_step in range(-self.mem_init_sz, self.train_steps + 1):
                if terminal:
                    state = env.reset()
                actions, mu, sigma = self.agent.policy(
                    np.expand_dims(state, 0),
                    training=True,
                    eps=eps.get() if global_step > 0 else 1.,
                    return_streams=True,
                )
                action = actions[0]
                if mu is not None and sigma is not None:
                    mu = mu.cpu()[0]
                    behavior.append(mu.argmax(0).item() != action)
                state, reward, terminal, lost_live = env.step(action)
                with self.mem_lock:
                    self.mem.put(state[-2], action, np.sign(reward), terminal or lost_live)
                if global_step < 0:
                    continue
                eps.step()

                if global_step % self.optimize_freq == 0:
                    try:
                        self.sync.get(block=True, timeout=10.)
                    except Empty:
                        continue
                if len(behavior) > 0:
                    if self.adaptive_eps is not None and global_step % self.adaptive_freq == 0:
                        real_eps = np.mean(behavior[-self.adaptive_freq:])
                        self.agent.c += 0.01 * np.sign(self.adaptive_eps - real_eps)
                        self.agent.c = max(0.01, self.agent.c)
                    if global_step % self.log_freq == 0:
                        if self.adaptive_eps is not None:
                            self.write(self.agent.c, "c", global_step)
                        self.write(np.mean(behavior), "behavior", global_step)
                        behavior = list()
        env.close()

    def train(self):
        Thread(target=self.play_thread,).start()
        self.sync.put(None)
        start_t = datetime.now()
        for global_step in range(0, self.train_steps + 1):

            if global_step % self.print_freq == 0:
                step_time = (datetime.now() - start_t) / self.print_freq
                start_t = datetime.now()
                print("every {} steps {}\t4M {}\t200M {}\tremain {}M,{}".format(
                    self.optimize_freq,
                    step_time * self.optimize_freq,
                    step_time * 10 ** 6,
                    step_time * (50 * 10 ** 6),
                    (self.train_steps - global_step) * 4 // 10 ** 6,
                    step_time * (self.train_steps - global_step),
                ))
            if global_step % self.update_target_freq == 0:
                self.agent.update_target()
            if global_step % self.eval_freq == 0:
                self.agent.update_eval()
                eval_thread = Thread(target=self.eval, args=(global_step,))
                eval_thread.start()

            if global_step % self.optimize_freq == 0:
                try:
                    self.sync.put(None, block=True, timeout=10.)
                except Full:
                    continue
                with self.mem_lock:
                    batch = self.mem.sample()
                idx, td_err = self.agent.optimize(*batch)
                if self.prioritized_replay:
                    with self.mem_lock:
                        self.mem.update_priority(idx, np.abs(td_err.cpu().numpy()))
        self.sync.task_done()
        eval_thread.join()
        return

    def eval(self, global_step):
        eval_func = dict(
            frames=self.eval_by_frames,
            episodes=self.eval_by_episodes,
        )[self.eval_method]
        reward = eval_func()
        self.write(reward, "reward", global_step)
        self.writer.flush()
        return

    def eval_by_episodes(self):
        n_trials = self.eval_episodes
        envs = [Atari(self.env_id) for _ in range(n_trials)]
        states = np.stack([u.reset() for u in envs])
        actions = np.empty(n_trials, dtype=np.int)
        reward = np.zeros(n_trials, dtype=np.float32)
        terminal = np.zeros(n_trials, dtype=np.bool)
        with torch.cuda.stream(self.cuda_eval):
            while not terminal.all():
                not_t = ~terminal
                actions[not_t] = self.agent.policy(
                    states=states[not_t],
                    training=False,
                    eps=self.eps_eval,
                    return_streams=False,
                )
                for i, nt in enumerate(not_t):
                    if nt:
                        states[i], r, terminal[i], _ = envs[i].step(actions[i])
                        reward[i] += r
        for e in envs:
            e.close()
        return np.mean(reward)

    def eval_by_frames(self):
        rewards = list()
        reward = 0.
        env = Atari(self.env_id)
        state = env.reset()
        with torch.cuda.stream(self.cuda_eval):
            for step in range(self.eval_frames // 4):
                action = self.agent.policy(
                    np.expand_dims(state, 0),
                    training=False,
                    eps=self.eps_eval,
                    return_streams=False,
                )[0]
                state, r, terminal, _ = env.step(action)
                reward += r
                if terminal:
                    rewards.append(reward)
                    reward = 0.
                    state = env.reset()
        env.close()
        return np.mean(rewards)

    def write(self, value, category, step):
        frm_idx = step * 4
        self.writer.add_scalars(
            main_tag="{}/{}".format(category, self.env_id),
            tag_scalar_dict={self.label: value},
            global_step=frm_idx,
        )
        if not os.path.exists(CSV_FOLDER):
            os.makedirs(CSV_FOLDER)
        path = os.path.join(
            CSV_FOLDER,
            "{}--{}--{}.csv".format(category, self.env_id, self.label),
        )
        has_header = os.path.exists(path)
        with open(path, "a") as fp:
            if not has_header:
                fp.write("frame (millions), {}\n".format(category))
            fp.write("{:.2f}, {:.3f}\n".format(frm_idx / 10 ** 6, value))
        return
