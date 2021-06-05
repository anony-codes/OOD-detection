import torch
import torch.nn as nn
import math

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class Ours(nn.Module):
    def __init__(self, w1, w2, w3, device):
        super(Ours, self).__init__()
        self.num_class = 10
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, in_feature, label):
        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)

        var_loss = 1 / (torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss = - torch.mean(distance) / math.sqrt(in_feature.size(1))

        corr = self.corr(in_feature.T) ** 2

        loss_corr = torch.mean(corr[torch.tril(torch.ones_like(corr), diagonal=-1) == 1])

        return self.w1 * mean_loss + self.w2 * var_loss + self.w3 * loss_corr

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return (1 / (D - 1)) * (X @ X.transpose(-1, -2))

    def corr(self, x):
        # calculate covariance matrix of rows
        c = self.cov(x)

        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c) + 1e-8)
        c = c.div(stddev.expand_as(c).t() + 1e-8)

        return c
