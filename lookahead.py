# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

import itertools as it
from copy import deepcopy
from torch.optim import Optimizer, Adam


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k

        for group in self.param_groups:
            group["step_counter"] = 0

        self.slow_weights = [
            [p.clone().detach() for p in group['params']]
                for group in self.param_groups
        ]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss

    def state_dict(self):
        d = deepcopy(self.optimizer.state_dict())
        d.update(dict(alpha=self.alpha, k=self.k))
        return d

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        self.k = state_dict['k']
        self.alpha = state_dict['alpha']
        del state_dict['k']
        del state_dict['alpha']
        self.optimizer.load_state_dict(state_dict)


def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
    adam = Adam(params, *args, **kwargs)
    return Lookahead(adam, alpha, k)
