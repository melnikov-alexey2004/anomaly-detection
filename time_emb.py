import torch
from torch import nn
import numpy as np
import math


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        # print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    # print(v1.shape)
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


from torch import nn
import torch
import typing
import math
import datetime


class ExtendedTimeEmb(nn.Module):
    def __init__(self, activation: typing.Literal["sin", "cos"], hid_dim):
        super().__init__()
        assert activation in ["sin", "cos"]
        if activation == "sin":
            self.l1 = SineActivation(1, hid_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hid_dim)



        self.fc1 = nn.Linear(hid_dim, 2)

    def cyclic_time_features(self, times:typing.Collection[datetime.datetime]):
        feats = []
        for t in times:
            h, m = t.hour / 24.0, t.minute / 60.0
            dow, mon = t.weekday() / 7.0, (t.month - 1) / 12.0
            dom = (t.day - 1) / 31.0
            feats.append([
                math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
                math.sin(2 * math.pi * m), math.cos(2 * math.pi * m),
                math.sin(2 * math.pi * dow), math.cos(2 * math.pi * dow),
                math.sin(2 * math.pi * mon), math.cos(2 * math.pi * mon),
                math.sin(2 * math.pi * dom), math.cos(2 * math.pi * dom),
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    sineact = SineActivation(1, 64)
    cosact = CosineActivation(1, 64)

    print(sineact(torch.Tensor([[7]])).requires_grad)
    print(sineact(torch.Tensor([[7]])).shape)
    print(sineact(torch.arange(1, 100).reshape((100, 1))).shape)
    print(cosact(torch.Tensor([[7]])).shape)
