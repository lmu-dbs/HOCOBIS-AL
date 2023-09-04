import numpy as np

from .strategy import Strategy


class RandomSampling(Strategy):
	def __init__(self, learner):
		super(RandomSampling, self).__init__(learner)

	def query(self, n):
		idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
		return np.random.choice(idxs_unlabeled, n, replace=False)
