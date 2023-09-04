import numpy as np
import torch
from .strategy import Strategy


class MarginSampling(Strategy):
    def __init__(self, learner):
        super(MarginSampling, self).__init__(learner)

    def query(self, n):
        idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
        probs = self.learner.predict(self.learner.X[idxs_unlabeled], self.learner.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        sorted_margin, sm_idx = U.sort()
        return idxs_unlabeled[sm_idx[:n]]


class EntropySampling(Strategy):
    def __init__(self, learner):
        super(EntropySampling, self).__init__(learner)

    def query(self, n):
        idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
        probs = self.learner.predict(self.learner.X[idxs_unlabeled], self.learner.Y[idxs_unlabeled])
        probs += 1e-8
        entropy = (probs * torch.log(probs)).sum(1)
        _, idx = entropy.sort()
        return idxs_unlabeled[idx[:n]]


class LCSampling(Strategy):
    def __init__(self, learner):
        super(LCSampling, self).__init__(learner)

    def query(self, n):
        idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
        probs = self.learner.predict(self.learner.X[idxs_unlabeled], self.learner.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0]
        sorted_lc, sm_idx = U.sort()
        return idxs_unlabeled[sm_idx[:n]]
