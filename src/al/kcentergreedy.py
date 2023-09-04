import numpy as np
import torch
from .strategy import Strategy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances


class KCenterGreedy(Strategy):
    def __init__(self, learner):
        super(KCenterGreedy, self).__init__(learner)

    def query(self, n):
        idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
        idxs_lb = np.arange(self.learner.n_pool)[self.learner.idxs_lb]

        embedding = self.learner.get_embedding(X=self.learner.X, Y=self.learner.Y)

        dist_mat = euclidean_distances(embedding[idxs_unlabeled], embedding[idxs_lb])
        min_dists = np.min(dist_mat, axis=-1)
        ind = min_dists.argmax()
        indsAll = [ind]
        features = [embedding[idxs_unlabeled[ind]]]
        while len(indsAll) < n:
            new_dist = pairwise_distances(embedding[idxs_unlabeled], features[-1].reshape(1, -1)).ravel().astype(float)
            for i in range(len(embedding[idxs_unlabeled])):
                if min_dists[i] > new_dist[i]:
                    min_dists[i] = new_dist[i]
            ind = min_dists.argmax()
            features.append(embedding[idxs_unlabeled[ind]])
            indsAll.append(ind)
        return idxs_unlabeled[indsAll]


class BalanceSampling(Strategy):
    def __init__(self, learner):
        super(BalanceSampling, self).__init__(learner)

    def query(self, n):
        idxs_unlabeled = np.arange(self.learner.n_pool)[~self.learner.idxs_lb]
        idxs_labeled = np.arange(self.learner.n_pool)[self.learner.idxs_lb]

        probs = self.learner.predict(self.learner.X[idxs_unlabeled], self.learner.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        confidence = probs_sorted[:, 0]
        pred = torch.max(probs, dim=-1)[1]  # n,

        _labels = self.learner.Y[idxs_labeled]
        diversity = torch.zeros_like(confidence)
        class_index, class_count = np.unique(_labels, return_counts=True)
        for c, count in zip(class_index, class_count):
            inverse_fraction_div = 1 - (count / len(_labels))
            mask = pred == c
            print(inverse_fraction_div)
            diversity[mask] = inverse_fraction_div
        diversity += confidence
        print(type(diversity))

        diversity_prob = diversity.numpy() / sum(diversity.numpy())
        # customDist = stats.rv_discrete(name='custm', values=(idxs_unlabeled, diversity_prob))
        # ind = customDist.rvs(size=n)
        print(sum(diversity_prob))
        ind = np.random.choice(idxs_unlabeled, size=n, p=diversity_prob, replace=False)

        return ind
