import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class PGPolicy(BasePolicy, nn.Module):
    """docstring for PGPolicy"""

    def __init__(self, model, optim, dist=Categorical,
                 discount_factor=0.99, normalized_reward=True):
        super().__init__()
        self.model = model
        self.optim = optim
        self.dist = dist
        self._eps = np.finfo(np.float32).eps.item()
        assert 0 <= discount_factor <= 1, 'discount_factor should in [0, 1]'
        self._gamma = discount_factor
        self._rew_norm = normalized_reward

    def process_fn(self, batch, buffer, indice):
        batch_size = len(batch.rew)
        returns = self._vanilla_returns(batch, batch_size)
        # returns = self._vectorized_returns(batch, batch_size)
        returns = returns - returns.mean()
        if self._rew_norm:
            returns = returns / (returns.std() + self._eps)
        batch.update(returns=returns)
        return batch

    def __call__(self, batch, state=None):
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        logits = F.softmax(logits, dim=1)
        dist = self.dist(logits)
        act = dist.sample().detach().cpu().numpy()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch, batch_size=None):
        losses = []
        for b in batch.split(batch_size):
            self.optim.zero_grad()
            dist = self(b).dist
            a = torch.tensor(b.act, device=dist.logits.device)
            r = torch.tensor(b.returns, device=dist.logits.device)
            loss = -(dist.log_prob(a) * r).sum()
            loss.backward()
            self.optim.step()
            losses.append(loss.detach().cpu().numpy())
        return losses

    def _vanilla_returns(self, batch, batch_size):
        returns = batch.rew[:]
        last = 0
        for i in range(batch_size - 1, -1, -1):
            if not batch.done[i]:
                returns[i] += self._gamma * last
            last = returns[i]
        return returns

    def _vectorized_returns(self, batch, batch_size):
        # according to my tests, it is slower than vanilla
        # import scipy.signal
        convolve = np.convolve
        # convolve = scipy.signal.convolve
        rew = batch.rew[::-1]
        gammas = self._gamma ** np.arange(batch_size)
        c = convolve(rew, gammas)[:batch_size]
        T = np.where(batch.done[::-1])[0]
        d = np.zeros_like(rew)
        d[T] += c[T] - rew[T]
        d[T[1:]] -= d[T[:-1]] * self._gamma ** np.diff(T)
        return (c - convolve(d, gammas)[:batch_size])[::-1]