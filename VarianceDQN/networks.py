import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):

    def __init__(self, in_dim, out_dim, std_init, device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        temp = torch.sqrt(torch.tensor(1. / in_dim, dtype=torch.float))
        self.w_mu = nn.Parameter(torch.empty((out_dim, in_dim), dtype=torch.float).uniform_(-temp, temp))
        self.b_mu = nn.Parameter(torch.empty(out_dim, dtype=torch.float).uniform_(-temp, temp))
        self.w_sigma = nn.Parameter(torch.empty((out_dim, in_dim), dtype=torch.float).fill_(std_init * temp))
        self.b_sigma = nn.Parameter(torch.empty(out_dim, dtype=torch.float).fill_(std_init * temp))
        del temp
        self.w_noise = None
        self.b_noise = None
        self.reset_noise()

    def generate_noise(self, n):
        noise = torch.randn(n, device=self.device)
        noise = noise.sign() * noise.abs().sqrt()
        return noise

    def reset_noise(self):
        self.w_noise = self.generate_noise(self.out_dim).ger(self.generate_noise(self.in_dim))
        self.b_noise = self.generate_noise(self.out_dim)

    def forward(self, x, reset_noise=False):
        if reset_noise:
            self.reset_noise()
        if self.training:
            w = self.w_mu + self.w_sigma * self.w_noise
            b = self.b_mu + self.b_sigma * self.b_noise
        else:
            w = self.w_mu
            b = self.b_mu
        return F.linear(x, w, b)


class DuelConvNet(nn.Module):

    def __init__(
            self,
            in_dims,
            out_dim,
            channels,
            kernels,
            strides,
            hidden_dim,
            device,
            uncertainty=False,
            noisy=False,
    ):
        super().__init__()
        self.device = device

        dims = [in_dims[0], *channels]
        layers = [nn.Conv2d(i, o, k, s) for i, o, k, s in zip(dims, dims[1:], kernels, strides)]
        layers = [u for c in layers for u in (c, nn.ReLU())]
        self.conv = nn.Sequential(*layers)

        def calc(_sz, _kernels, _strides):
            for _k, _s in zip(_kernels, _strides):
                _sz = (_sz - _k) // _s + 1
            return _sz

        self.noisy = noisy
        if noisy:
            linear_cls = NoisyLinear
            additional = dict(std_init=.5, device=device)
        else:
            linear_cls = nn.Linear
            additional = dict()
        in_dim = calc(in_dims[1], kernels, strides) * calc(in_dims[2], kernels, strides) * channels[-1]
        self.value1 = linear_cls(in_dim, hidden_dim, **additional)
        self.value2 = linear_cls(hidden_dim, 1, **additional)
        self.advantage1 = linear_cls(in_dim, hidden_dim, **additional)
        self.advantage2 = linear_cls(hidden_dim, out_dim, **additional)

        self.uncertainty = uncertainty
        if uncertainty:
            self.sigma = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x, reset_noise=False):
        additional = dict(reset_noise=True) if self.noisy and reset_noise else dict()
        x /= 255.
        x = self.conv(x)
        common = x.view(x.size(0), -1)
        value = self.value1(common, **additional)
        value = torch.relu(value)
        value = self.value2(value, **additional)
        advantage = torch.relu(self.advantage1(common, **additional))
        advantage = self.advantage2(advantage, **additional)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        if self.training and self.uncertainty:
            sigma = self.sigma(common)
            return q, sigma
        else:
            return q, None


class ConvNet(nn.Module):

    def __init__(
            self,
            in_dims,
            out_dim,
            channels,
            kernels,
            strides,
            hidden_dim,
            device,
            uncertainty=False,
            noisy=False,
    ):
        super().__init__()
        self.device = device

        dims = [in_dims[0], *channels]
        layers = [nn.Conv2d(i, o, k, s) for i, o, k, s in zip(dims, dims[1:], kernels, strides)]
        layers = [u for c in layers for u in (c, nn.ReLU())]
        self.conv = nn.Sequential(*layers)

        def calc(_sz, _kernels, _strides):
            for _k, _s in zip(_kernels, _strides):
                _sz = (_sz - _k) // _s + 1
            return _sz

        self.noisy = noisy
        if noisy:
            linear_cls = NoisyLinear
            additional = dict(std_init=.5, device=device)
        else:
            linear_cls = nn.Linear
            additional = dict()
        in_dim = calc(in_dims[1], kernels, strides) * calc(in_dims[2], kernels, strides) * channels[-1]
        self.linear1 = linear_cls(in_dim, hidden_dim, **additional)
        self.linear2 = linear_cls(hidden_dim, out_dim, **additional)

        self.uncertainty = uncertainty
        if uncertainty:
            self.sigma = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x, reset_noise=False):
        additional = dict(reset_noise=True) if self.noisy and reset_noise else dict()
        x /= 255.
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        linear1 = torch.relu(self.linear1(x, **additional))
        q = self.linear2(linear1, **additional)
        if self.training and self.uncertainty:
            sigma = self.sigma(x)
            return q, sigma
        else:
            return q, None

