import torch
from torch.nn import functional as F
import numpy as np

from networks import DuelConvNet, ConvNet


class DQNAgent:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        net_cls = DuelConvNet if self.duel else ConvNet
        net_kwargs = dict(
            in_dims=(self.history_len, *self.state_shape),
            out_dim=self.n_actions,
            channels=self.channels,
            kernels=self.kernels,
            strides=self.strides,
            hidden_dim=self.hidden_dim,
            device=self.device,
            uncertainty=self.uncertainty,
            noisy=self.noisy,
        )
        self.q = net_cls(**net_kwargs).to(self.device)
        optim_cls = dict(
            adam=torch.optim.Adam,
            rmsprop=torch.optim.RMSprop,
        )[self.optim]
        self.optimizer = optim_cls(
            self.q.parameters(),
            **self.optim_kwargs,
        )
        self.q_ = net_cls(**net_kwargs).to(self.device)
        self.q_eval = net_cls(**net_kwargs).to(self.device)
        self.update_target()
        if self.duel or self.uncertainty:
            self.grad_scalar = torch.sqrt(
                torch.tensor(1. / (1. + float(self.duel) + float(self.uncertainty)),
                             dtype=torch.float,
                             device=self.device)
            )

    def update_target(self):
        self.q_.load_state_dict(self.q.state_dict())

    def update_eval(self):
        self.q_eval.load_state_dict(self.q.state_dict())
        self.q_eval.eval()

    def policy(self, states, training, eps, return_streams=False):
        n = states.shape[0]
        actions = np.zeros(n, dtype=np.int)
        random = np.random.random(n) < eps
        perform = np.logical_not(random)
        actions[random] = np.random.randint(0, self.n_actions, np.sum(random))
        if perform.any():
            net = self.q if training else self.q_eval
            with torch.no_grad():
                mu, sigma = net(
                    torch.from_numpy(states[perform]).to(self.device).float(),
                )
                if training and self.uncertainty:
                    sigma = sigma.abs()
                    q = mu + sigma * self.c
                else:
                    q = mu
                actions[perform] = q.argmax(1).cpu().numpy()
        else:
            mu, sigma = None, None
        if return_streams:
            return actions, mu, sigma
        else:
            return actions

    def optimize(self, state, state_, action, reward, terminal, is_weight=None, idx=None):
        q_ = reward
        not_t = ~terminal
        action = action.unsqueeze(1)
        q, sigma = self.q(state, reset_noise=True)
        q = q.gather(1, action).squeeze(1)
        with torch.no_grad():
            q_next = self.q_(state_[not_t], reset_noise=True)[0]
            if self.double:
                action_ = self.q(state_[not_t])[0].argmax(1, keepdim=True)
                q_next = q_next.gather(1, action_).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
            q_[not_t] += q_next.detach() * self.gamma
            td_err = (q_ - q).detach().abs()
        if self.uncertainty:
            sigma = sigma.gather(1, action).squeeze(1)
        loss = F.smooth_l1_loss(q, q_, reduction="none")
        if self.uncertainty:
            if self.method == "TDDQN":
                loss = loss + F.smooth_l1_loss(sigma, td_err, reduction="none")
            elif self.method == "VDQN":
                loss = loss + F.smooth_l1_loss(sigma.pow(2), td_err.pow(2), reduction="none")
        if self.prioritized_replay:
            loss = loss * is_weight
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.duel or self.uncertainty:
            for param in list(self.q.parameters())[:6]:
                param.grad *= self.grad_scalar
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.)
        self.optimizer.step()
        return idx, td_err.clamp(1.)
