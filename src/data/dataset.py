import os

import numpy as np

from sklearn.cluster import KMeans
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import ddu_dirty_mnist

from src.settings import DATA_ROOT, DIRTY_MNIST_CONFIG


def create_imbalance(X, Y, class_index, imb_classes=0.5, imbalance_ratio=0.9):
    """
    Creates imbalanced version of a given dataset with inputs X and targets Y.
    :param X: input
    :param Y: target
    :param class_index: class indices
    :param imb_classes: fraction of classes that will be imbalanced
    :param imbalance_ratio: fraction of instances that will be removed from imbalanced classes
    :return: imbalanced X, imbalanced Y
    """
    random_classes = np.random.choice(range(len(class_index)), size=int(len(class_index) * imb_classes),
                                      replace=False)
    for i, c in enumerate(random_classes):
        idx_c = np.arange(len(Y))[Y == c]  # idx of samples with class c
        n_remove = int(len(idx_c) * imbalance_ratio)  # number of samples to remove
        rand_remove = np.random.choice(range(len(idx_c)), size=n_remove, replace=False)
        assert all(Y[idx_c[rand_remove]] == c)
        if i == 0:
            indices_to_remove = idx_c[rand_remove]  # map back to original indices
        else:
            indices_to_remove = np.concatenate([indices_to_remove, idx_c[rand_remove]]).ravel()
    tr_y_imb = np.delete(Y, indices_to_remove, axis=0)
    tr_x_imb = np.delete(X, indices_to_remove, axis=0)
    return tr_x_imb, tr_y_imb


def create_repeated(X, Y, repetitions=3):
    """
    Creates repeated version of a given dataset and applies random gaussian noise on each repeated instance
    to get slight differences between duplicates.
    :param X: input
    :param Y: target
    :param repetitions: number of times each instance will be copied
    :return: repeated X, repeated Y
    """
    shape = X.shape
    original_length = shape[0]
    X_tr = X.repeat(repetitions, 1, 1)
    Y_tr = Y.repeat(repetitions)
    n_delete = original_length * (repetitions-1)
    rand_remove = np.random.choice(range(original_length*repetitions), size=n_delete, replace=False)

    for rep in range(1, repetitions, 1):
        dataset_noise = torch.empty(shape, dtype=torch.float32).normal_(0.0, 0.1)
        start_idx = int(rep*original_length)
        end_idx = int((rep*original_length)+original_length)
        X_tr[start_idx:end_idx] = X_tr[start_idx:end_idx] + dataset_noise

    tr_y_rep = np.delete(Y_tr, rand_remove, axis=0)
    tr_x_rep = np.delete(X_tr, rand_remove, axis=0)
    print(len(tr_y_rep))
    return tr_x_rep, tr_y_rep


# training and data settings for dataset
def get_transform(name):
    if name == 'MNIST' or or name == "ImbalancedMNIST" or name == "IntraClassImbalanceMNIST":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif name == 'DirtyMNIST':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(name, path=DATA_ROOT, imbalanced=False):
    if name == 'MNIST':
        return get_MNIST(path, imbalanced=imbalanced)
    elif name == 'DirtyMNIST':
        return get_DirtyMNIST(path, imbalanced=imbalanced)
    elif name == "IntraClassImbalanceMNIST":
        return get_IntraClassImbalanceMNIST(path)


def get_MNIST(path, imbalanced=False):
    raw_tr = datasets.MNIST(path, train=True, download=True)
    raw_te = datasets.MNIST(path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets

    if imbalanced:
        class_index, class_count = np.unique(Y_tr, return_counts=True)
        X_tr, Y_tr = create_imbalance(X_tr, Y_tr, class_index)

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_DirtyMNIST(path, imbalanced=False):
    raw_tr = ddu_dirty_mnist.DirtyMNIST(path, train=True, download=True, normalize=False, noise_stddev=0.0015, transform=None)
    raw_te = ddu_dirty_mnist.DirtyMNIST(path, train=False, download=True, normalize=False, transform=None)
    _amb_x = raw_tr.datasets[1].data
    _amb_y = raw_tr.datasets[1].targets

    _clean_x = raw_tr.datasets[0].data
    _clean_y = raw_tr.datasets[0].targets

    n_amb_tr = round(len(_amb_x) * DIRTY_MNIST_CONFIG["amb_proportion_tr"])
    n_clean_tr = round(len(_clean_x) * (1-DIRTY_MNIST_CONFIG["amb_proportion_tr"]))

    n_amb_te = round(len(raw_te.datasets[1].data) * DIRTY_MNIST_CONFIG["amb_proportion_te"])

    dirty_idx = np.arange(len(_amb_x))  # all points in pool False = not labelled
    np.random.shuffle(dirty_idx)  # randomly shuffle
    dirty_data, dirty_targets = _amb_x[dirty_idx[:n_amb_tr]], _amb_y[dirty_idx[:n_amb_tr]]

    clean_idx = np.arange(len(_clean_x))  # all points in pool False = not labelled
    np.random.shuffle(clean_idx)
    clean_data, clean_targets = _clean_x[clean_idx[:n_clean_tr]], _clean_y[clean_idx[:n_clean_tr]]

    ## train
    X_tr = torch.cat((clean_data, dirty_data))
    Y_tr = torch.cat((clean_targets, dirty_targets))

    ## test
    if n_amb_te > 0:
        X_te = torch.cat((raw_te.datasets[0].data, raw_te.datasets[1].data[:n_amb_te]))
        Y_te = torch.cat((raw_te.datasets[0].targets, raw_te.datasets[1].targets[:n_amb_te]))
    else:
        X_te = raw_te.datasets[0].data
        Y_te = raw_te.datasets[0].targets

    if imbalanced:
        # create imbalance -> remove large fraction of datapoints from some classes
        class_index, class_count = np.unique(Y_tr, return_counts=True)
        X_tr, Y_tr = create_imbalance(X_tr, Y_tr, class_index)

    X_tr = X_tr * 255
    X_tr = X_tr.to(torch.uint8)
    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)

    tr_x_dirty = X_tr
    tr_y_dirty = Y_tr
    print(len(tr_y_dirty))
    return tr_x_dirty, tr_y_dirty, X_te, Y_te, N


def get_IntraClassImbalanceMNIST(path):
    name_images = "IntraClassImbalanceImages.pt"
    name_labels = "IntraClassImbalanceLabels.pt"
    raw_te = datasets.MNIST(path, train=False, download=True)
    if os.path.exists(os.path.join(path, name_images)):
        X_tr = torch.load(os.path.join(path, name_images))
        Y_tr = torch.load(os.path.join(path, name_labels))
    else:
        raw_tr = datasets.MNIST(path, train=True, download=True)
        Y_tr = raw_tr.targets
        X_tr = raw_tr.data.reshape(60000, -1)

        class_index, class_count = np.unique(Y_tr, return_counts=True)
        label_masks = []
        clusters_per_class = []

        images_idx = np.arange(len(Y_tr))
        clusters = 500

        for c, count in zip(class_index, class_count):
            mask = Y_tr == c
            label_masks.append(images_idx[mask])  # idx per class
            cluster_learner = KMeans(n_clusters=clusters)
            cluster_learner.fit(X_tr[mask])
            cluster_idxs = cluster_learner.predict(X_tr[mask])
            clusters_per_class.append(cluster_idxs)

        major_class = [np.unique(c, return_counts=True) for c in clusters_per_class]

        major_class = [(np.max(mc[1]), np.argmax(mc[1])) for mc in major_class]
        keep_final = []
        all_images = None
        all_labels = None

        for i, (c, count) in enumerate(zip(class_index, class_count)):
            class_mask = (Y_tr   == c)
            # mask holding idx with
            class_cluster_mask = clusters_per_class[i] != major_class[i][1]
            min_clusters = np.arange(sum(class_mask))[class_cluster_mask]
            keep_ind = np.random.choice(min_clusters, size=clusters - 1, replace=False)
            remain = sum(class_mask) - (major_class[i][0] + clusters - 1)
            major_class_idx = label_masks[i][~class_cluster_mask]
            remaining_class_idx = label_masks[i][keep_ind]
            keep_final.append(np.concatenate((major_class_idx, remaining_class_idx),
                                             axis=None))
            image_features_major = X_tr[major_class_idx]
            image_features_remain = X_tr[remaining_class_idx]
            repetitions = int(remain / major_class[i][0])  # remain: Anzahl an samples die fehlen

            image_features_major = image_features_major.reshape(-1, 28, 28)
            image_features_remain = image_features_remain.reshape(-1, 28, 28)
            X_new = image_features_major.repeat(repetitions, 1, 1)
            Y_new = Y_tr[major_class_idx].repeat(repetitions)
            Y_remain = Y_tr[remaining_class_idx]
            shape = image_features_major.shape

            original_length = shape[0]
            for rep in range(1, repetitions, 1):
                dataset_noise = torch.empty(shape, dtype=torch.float32).normal_(0.0, 0.1)
                start_idx = int(rep * original_length)
                end_idx = int((rep * original_length) + original_length)
                X_new[start_idx:end_idx] = X_new[start_idx:end_idx] + dataset_noise

            if all_images is None:
                all_images = np.concatenate((X_new, image_features_remain), axis=0)
                all_labels = np.concatenate((Y_new, Y_remain), axis=0)
            else:
                bla = np.concatenate((X_new, image_features_remain), axis=0)
                all_images = np.concatenate((all_images, bla), axis=0)
                labelsbla = np.concatenate((Y_new, Y_remain), axis=0)
                all_labels = np.concatenate((all_labels, labelsbla), axis=0)

        X_tr = torch.as_tensor(all_images)
        Y_tr = torch.as_tensor(all_labels)
        torch.save(X_tr, os.path.join(path, name_images))
        torch.save(Y_tr, os.path.join(path, name_labels))

    X_te = raw_te.data
    Y_te = raw_te.targets
    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_handler(name):
    if name == 'MNIST' or name == "ImbalancedMNIST" or name == "IntraClassImbalanceMNIST":
        return MNISTHandler
    elif name == 'DirtyMNIST':
        return DirtyMNISTHandler
    else:
        print("Dataset not found")


class MNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy())
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DirtyMNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.reshape(-1, 28)
            x = Image.fromarray(x.numpy())
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
