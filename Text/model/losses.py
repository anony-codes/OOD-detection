import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
# from pytorch.util.loss_utils import *

import math
from overrides import overrides


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class BCELoss(_Loss):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criteria = nn.BCELoss()
        self.one = torch.FloatTensor([1]).cuda()
        self.zero = torch.FloatTensor([0]).cuda()

    def forward(self, in_feature, label):
        """
        in_feature (N, hidden_size)
        """

        self.criteria(in_feature, label)

        return self.criteria(in_feature, label)

    @torch.no_grad()
    def predict(self, logits):
        logits = logits.view(-1)
        out = torch.where(logits >= 0.5, self.one, self.zero)

        return out.view(-1), out


class BaseLoss(_Loss):  # inner product
    def __init__(self, args, fc: nn.Linear, num_class=2):
        super(BaseLoss, self).__init__()
        self.args = args
        self.fc = fc  # (hidden,n_class)
        self.w = self.fc.weight  # self.w (n_class,hidden_size)
        self.b = self.fc.bias  # n_class
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_class = num_class

    def forward(self, in_feature, label):
        """
        in_feature (N, hidden_size)
        """

        logits = self.fc(in_feature.to(self.fc.weight.dtype))


        return self.loss_func(torch.log_softmax(logits, -1), label)

    @torch.no_grad()
    def predict(self, in_feature, mu_matrix=None):
        in_feature = in_feature.to(torch.float32)

        logits = self.fc(in_feature.to(self.fc.weight.dtype))

        return torch.argmax(logits, dim=-1), torch.softmax(logits, -1),


class Regularizer(BaseLoss):
    def __init__(self, args, fc, lambda_mu=0.3, lambda_var=None, lambda_corr=None):
        super(Regularizer, self).__init__(args, fc)
        self.lambda_mu = lambda_mu
        self.zero = torch.FloatTensor([0.0]).to(self.fc.weight.device)
        self.lambda_var = lambda_var
        self.lambda_corr = lambda_corr
        self.basic_loss = BaseLoss(args, fc)

    @overrides
    def forward(self, in_feature, label):
        """\
        in_feature : (N, hidden_size) torch.float32
        label : (N) torch.int64
        """

        hidden_size = in_feature.size(-1)
        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)


        basic_loss = self.basic_loss(in_feature, label)

        ## dist loss
        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        dist_loss = - torch.mean(distance) / math.sqrt(hidden_size)

        ## entropy loss
        var_loss = torch.tensor(1.).to('cuda') / (torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        # var_loss = torch.mean(1 / (torch.var(in_feature, dim=0) + (1e-6)))
        corr = self.corr(in_feature.T) ** 2
        loss_corr = torch.mean(torch.tril(corr, -1))


        return basic_loss + self.lambda_mu * dist_loss + self.lambda_var * var_loss + self.lambda_corr * loss_corr


    def corr(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        cov = 1 / (D - 1) * X @ X.transpose(-1, -2)

        inv = torch.inverse(torch.diag(torch.diag(cov)).sqrt())
        r = inv @ cov @ inv

        return r

    @overrides
    def predict(self, in_feature, mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        logits = self.fc(in_feature.to(self.fc.weight.dtype))
        return torch.argmax(logits, dim=-1), torch.softmax(logits, dim=-1)

    def predict_dist(self, in_feature, mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        dist = torch.cdist(in_feature, mu_matrix)
        # values, indices = torch.max(-dist, dim=-1)

        return torch.argmin(dist, dim=-1), -dist


class SSDLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SSDLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        #         normalize hidden representation

        l2_norm = torch.norm(features, dim=-1, p=2).unsqueeze(-1)
        features = features / l2_norm

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
