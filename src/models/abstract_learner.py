import copy
import math
from collections import Counter

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from src.models.utils import adjust_learning_rate, warmup_learning_rate
from src.data import transforms as data_transforms


class AbstractLearner:
    def __init__(self, backbone: nn.Module, backbone_args: dict, X, Y, idxs_lb, args: dict, handler):
        super().__init__()
        self.backbone = backbone
        self.backbone_args = backbone_args

        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb

        self.train_params = args["train_params"]
        self.ds_params = args["ds_params"]

        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.current_round = 0
        self.total_rounds = 0
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.patience, self.frequency = 10, 5

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, optimizer):
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.ds_params['transform']),
            shuffle=True,
            **self.ds_params['loader_tr_args']
        )

        self.clf.train()
        train_accuracy = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = self.loss_fc(out, y)

            train_accuracy += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_accuracy /= len(loader_tr.dataset.X)
        train_loss /= len(loader_tr.dataset.X)
        return train_accuracy, train_loss

    def train(self):
        self.clf = self.backbone(**self.backbone_args).to(self.device)
        early_stopping = False
        if 'early_stopping' in self.train_params:
            early_stopping = self.train_params["early_stopping"]

        optimizer = optim.Adam(
            self.clf.parameters(),
            lr=self.train_params['lr'],
            weight_decay=self.train_params['wd']
        )

        loss_hist, acc_hist = [], []
        tr_loss, tr_acc = 0, 0
        # outer loop
        for epoch in range(self.train_params["epochs"]):
            tr_acc, tr_loss = self._train(optimizer)
            loss_hist.append(tr_loss)
            acc_hist.append(tr_acc)
            print(f'Epoch {epoch:5}: {tr_loss:2.7f} (acc: {tr_acc})')

            if early_stopping and epoch > self.patience:
                if tr_acc >= 0.99:
                    print('Early Stopping.')
                    break
        return tr_acc, tr_loss

    def predict(self, X, Y, prob=True):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.ds_params['transform']),
            shuffle=False,
            **self.ds_params['loader_te_args']
        )
        self.clf.eval()
        vals = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                vals[idxs] = F.softmax(out, dim=1).cpu() if prob else out.cpu()
        return vals

    def get_embedding(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding

    def predict_label(self, X, Y):
        predictions = self.predict(X, Y)
        P = torch.max(predictions, dim=-1)[1]
        return P

    @staticmethod
    def loss_fc(soft_output_pred, soft_output):
        return F.cross_entropy(soft_output_pred, soft_output)


class Supervised(AbstractLearner):
    def __init__(self, backbone: nn.Module, backbone_args: dict, X, Y, idxs_lb, args: dict, handler):
        super().__init__(backbone, backbone_args, X, Y, idxs_lb, args, handler)


class PseudoLabel(AbstractLearner):
    def __init__(self, backbone: nn.Module, backbone_args: dict, X, Y, idxs_lb, args: dict, handler):
        super().__init__(backbone, backbone_args, X, Y, idxs_lb, args, handler)
        self.threshold = 0.95

    def _train(self, optimizer):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(
            self.handler(self.X, self.Y, transform=self.ds_params['transform']),
            shuffle=True,
            **self.ds_params['loader_tr_args']
        )

        assert len(idxs_unlabeled) + len(idxs_train) == len(loader_tr.dataset.X)
        self.clf.train()
        train_accuracy = 0.
        train_loss = 0.
        n_pl = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            lb_mask_batch = copy.deepcopy(idxs)
            pl_mask_batch = copy.deepcopy(idxs)
            lb_mask_batch = lb_mask_batch.apply_(lambda x: x in idxs_train).bool()
            pl_mask_batch = pl_mask_batch.apply_(lambda x: x in idxs_unlabeled).bool()

            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            lb_loss = 0
            if len(x[lb_mask_batch]) > 0:
                out, e1 = self.clf(x[lb_mask_batch])
                lb_loss = self.loss_fc(out, y[lb_mask_batch])

                train_accuracy += torch.sum((torch.max(out, 1)[1] == y[lb_mask_batch]).float()).data.item()
            pl_loss = 0
            if len(x[pl_mask_batch]) > 0:
                out_pl, e1_pl = self.clf(x[pl_mask_batch])
                pred = F.softmax(out_pl, dim=1)
                pred_max, pred_argmax = torch.max(pred, dim=-1)
                pl_threshold_mask = pred_max > self.threshold
                n_pl += sum(pl_threshold_mask)
                if any(pl_threshold_mask):
                    # print(pl_threshold_mask)
                    # print(pred_max[pl_threshold_mask])
                    # print(pred_argmax[pl_threshold_mask])
                    pl_loss = self.loss_fc(out_pl[pl_threshold_mask], pred_argmax[pl_threshold_mask])

            loss = lb_loss + pl_loss
            if loss > 0:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_accuracy /= len(idxs_train)
        train_loss /= len(idxs_train) + n_pl
        return train_accuracy, train_loss


class FixMatch(AbstractLearner):
    def __init__(self, backbone: nn.Module, backbone_args: dict, X, Y, idxs_lb, args: dict, handler):
        super().__init__(backbone, backbone_args, X, Y, idxs_lb, args, handler)
        self.warmup_epoch = 0
        self.schedule = [120, 160]
        self.lambda_u = 10
        self.threshold = 0.7
        self.momentum = 0.9
        self.nesterov = False
        self.wd = 1e-4

    def train(self):
        self.clf = self.backbone(**self.backbone_args).to(self.device)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_ul = np.arange(self.n_pool)[~self.idxs_lb]
        # loader_tr = DataLoader(
        #     self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.ds_params['transform']),
        #     shuffle=True,
        #     **self.ds_params['loader_tr_args']
        # )
        train_dataset_x, train_dataset_u = get_imagenet_ssl(
            handler=self.handler,
            appendix_augs=self.ds_params['transform'],
            X=self.X, Y=self.Y, trainindex_x=idxs_train, trainindex_u=idxs_ul,
            weak_type="DefaultTrain", strong_type="RandAugment")

        train_sampler = RandomSampler
        train_loader_x = DataLoader(
            train_dataset_x,
            sampler=train_sampler(train_dataset_x),
            batch_size=self.ds_params['loader_tr_args']['batch_size'],
            num_workers=self.ds_params['loader_tr_args']['num_workers'])

        train_loader_u = DataLoader(
            train_dataset_u,
            sampler=train_sampler(train_dataset_u),
            batch_size=self.ds_params['loader_tr_args']['batch_size'] * 5,
            num_workers=self.ds_params['loader_tr_args']['num_workers'])

        early_stopping = False
        if 'early_stopping' in self.train_params:
            early_stopping = self.train_params["early_stopping"]

        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=self.train_params['lr'],
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=self.nesterov
        )

        loss_hist, acc_hist = [], []
        tr_loss, tr_acc = 0, 0
        # outer loop
        for epoch in range(self.train_params["epochs"]):
            if epoch >= self.warmup_epoch:
                # lr, warmup_epoch, epochs, schedule, cos
                adjust_learning_rate(
                    optimizer, epoch, self.train_params['lr'],
                    self.warmup_epoch, self.train_params["epochs"],
                    self.schedule, cos=False
                )

            tr_loss, tr_acc = self._train(train_loader_x, train_loader_u, optimizer, epoch)
            loss_hist.append(tr_loss)
            acc_hist.append(tr_acc)
            print(f'Epoch {epoch:5}: {tr_loss:2.7f} (acc: {tr_acc})')
            if early_stopping and epoch > self.patience:
                if tr_acc >= 0.99:
                    print('Early Stopping.')
                    break
        return tr_acc, tr_loss

    def _train(self, train_loader_x, train_loader_u, optimizer, epoch):
        self.clf.train()
        epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
        train_accuracy = 0.
        train_loss = 0.
        total_num = 0.0
        num_train = 0
        train_iter_x = iter(train_loader_x)

        for i, (images_u, targets_u, idx_u) in enumerate(train_loader_u):
            try:
                images_x, targets_x, idx_x = next(train_iter_x)
            except Exception:
                epoch_x += 1
                # print("reshuffle train_loader_x at epoch={}".format(epoch_x))
                train_iter_x = iter(train_loader_x)
                images_x, targets_x, idx_x = next(train_iter_x)
            images_u_w, images_u_s = images_u
            num_batch = images_x.shape[0] + images_u_w.shape[0]
            total_num += num_batch
            # warmup learning rate
            if epoch < self.warmup_epoch:
                warmup_step = self.warmup_epoch * len(train_loader_u)
                curr_step = epoch * len(train_loader_u) + i + 1
                warmup_learning_rate(optimizer, curr_step, warmup_step, self.train_params['lr'])
            # curr_lr.update(optimizer.param_groups[0]['lr'])

            # model forward
            inputs = torch.cat((images_x, images_u_w, images_u_s))
            images_x, targets_x = images_x.to(self.device), targets_x.to(self.device)
            images_u_w, images_u_s = images_u_w.to(self.device), images_u_s.to(self.device)
            logits_x, logits_u_w, logits_u_s = self.clf(images_x, images_u_w, images_u_s)
            # pseudo label
            pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
            max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float()

            # compute losses
            loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
            loss_u = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none') * mask).mean()
            loss = loss_x + self.lambda_u * loss_u

            train_accuracy += torch.sum((torch.max(logits_x, 1)[1] == targets_x).float()).data.item()
            num_train += len(targets_x)
            train_loss += loss.item()
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.clf.momentum_update_ema()
        train_loss /= total_num
        train_accuracy /= num_train
        return train_loss, train_accuracy


# Credits to https://github.com/TorchSSL/TorchSSL
class FlexMatch(AbstractLearner):
    def __init__(self, backbone: nn.Module, backbone_args: dict, X, Y, idxs_lb, args: dict, handler):
        super().__init__(backbone, backbone_args, X, Y, idxs_lb, args, handler)
        self.warmup_epoch = 0
        self.schedule = [120, 160]
        self.lambda_u = 10
        self.threshold = 0.7
        self.momentum = 0.9
        self.nesterov = False
        self.wd = 1e-4
        self.amp = False
        self.thresh_warmup = True
        self.clip = 0
        T=0.5
        self.t_fn = Get_Scalar(T)  # temperature params function
        p_cutoff=0.95
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.hard_label = True
        self.it = 0

    def _train(self, train_loader_x, train_loader_u, optimizer, epoch):

        self.clf.train()

        train_accuracy = 0.
        train_loss = 0.
        total_num = 0.0
        num_train = 0
        train_iter_x = iter(train_loader_x)

        p_model = None


        selected_label = torch.ones((self.len_ul,), dtype=torch.long, ) * -1
        # print(f"len selected label: {len(selected_label)}")
        selected_label = selected_label.to(self.device)

        classwise_acc = torch.zeros((10,)).to(self.device)

        for i, (x_ulb, y_ulb, x_ulb_idx) in enumerate(train_loader_u):
            # print(f"x_ulb_idx: {x_ulb_idx}")
            # print(f"x_ulb_idx.shape: {x_ulb_idx.shape}")
            try:
                x_lb, y_lb, idx_x = next(train_iter_x)
            except Exception:
                train_iter_x = iter(train_loader_x)
                x_lb, y_lb, idx_x = next(train_iter_x)
            x_ulb_w, x_ulb_s = x_ulb
            num_batch = x_lb.shape[0] + x_ulb_w.shape[0]
            total_num += num_batch

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.to(self.device), x_ulb_w.to(self.device), x_ulb_s.to(self.device)
            x_ulb_idx = x_ulb_idx.to(self.device)
            y_lb = y_lb.to(self.device)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(train_loader_u):  # not all(5w) -1
                if self.thresh_warmup:
                    for i in range(10):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = copy.deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(10):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            # inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            # with amp_cm():
            # logits = self.clf(inputs)
            logits_x_lb, logits_x_ulb_w, logits_x_ulb_s = self.clf(x_lb, x_ulb_w, x_ulb_s)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # hyper-params for update
            T = self.t_fn(self.it)
            p_cutoff = self.p_fn(self.it)

            unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(
                logits_x_ulb_s,
                logits_x_ulb_w,
                classwise_acc,
                None,
                p_model,
                'ce', T, p_cutoff,
                use_hard_labels=self.hard_label,
                use_DA=False
            )

            if x_ulb_idx[select == 1].nelement() != 0:
                selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            total_loss = sup_loss + self.lambda_u * unsup_loss
            train_accuracy += torch.sum((torch.max(logits_x_lb, 1)[1] == y_lb).float()).data.item()
            num_train += len(y_lb)
            train_loss += total_loss.item()

            total_loss.backward()
            if (self.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.clf.parameters(), self.clip)
            optimizer.step()

            self.clf.zero_grad()
            self.it += 1
        train_loss /= total_num
        train_accuracy /= num_train
        return train_loss, train_accuracy

    def train(self):
        self.clf = self.backbone(**self.backbone_args).to(self.device)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_ul = np.arange(self.n_pool)[~self.idxs_lb]
        self.len_ul=len(idxs_ul)

        train_dataset_x, train_dataset_u = get_imagenet_ssl(
            handler=self.handler,
            appendix_augs=self.ds_params['transform'],
            X=self.X, Y=self.Y, trainindex_x=idxs_train, trainindex_u=idxs_ul,
            weak_type="DefaultTrain", strong_type="RandAugment")

        train_sampler = RandomSampler
        train_loader_x = DataLoader(
            train_dataset_x,
            sampler=train_sampler(train_dataset_x),
            batch_size=self.ds_params['loader_tr_args']['batch_size'],
            num_workers=self.ds_params['loader_tr_args']['num_workers'])

        train_loader_u = DataLoader(
            train_dataset_u,
            sampler=train_sampler(train_dataset_u),
            batch_size=self.ds_params['loader_tr_args']['batch_size'] * 5,
            num_workers=self.ds_params['loader_tr_args']['num_workers'])

        early_stopping = False
        if 'early_stopping' in self.train_params:
            early_stopping = self.train_params["early_stopping"]

        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=self.train_params['lr'],
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=self.nesterov
        )

        loss_hist, acc_hist = [], []
        tr_loss, tr_acc = 0, 0
        # outer loop
        for epoch in range(self.train_params["epochs"]):
            if epoch >= self.warmup_epoch:
                # lr, warmup_epoch, epochs, schedule, cos
                adjust_learning_rate(
                    optimizer, epoch, self.train_params['lr'],
                    self.warmup_epoch, self.train_params["epochs"],
                    self.schedule, cos=False
                )

            tr_loss, tr_acc = self._train(train_loader_x, train_loader_u, optimizer, epoch)
            loss_hist.append(tr_loss)
            acc_hist.append(tr_acc)
            print(f'Epoch {epoch:5}: {tr_loss:2.7f} (acc: {tr_acc})')
            if early_stopping and epoch > self.patience:
                if tr_acc >= 0.99:
                    print('Early Stopping.')
                    break
        return tr_acc, tr_loss


def get_imagenet_ssl(handler, appendix_augs, X, Y, trainindex_x, trainindex_u,
                     train_type='DefaultTrain', weak_type='DefaultTrain',
                     strong_type='RandAugment'):
    transform_x = data_transforms.get_transforms(train_type, appendix_augs)
    weak_transform = data_transforms.get_transforms(weak_type, appendix_augs)
    strong_transform = data_transforms.get_transforms(strong_type, appendix_augs)
    transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    train_dataset_x = handler(X[trainindex_x], Y[trainindex_x], transform=transform_x)
    train_dataset_u = handler(X[trainindex_u], Y[trainindex_u], transform=transform_u)
    return train_dataset_x, train_dataset_u


def consistency_loss(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), p_model

    else:
        assert Exception('Not Implemented consistency_loss')


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value
