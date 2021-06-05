import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import math
# from pytorch.util.loss_utils import *
import numpy as np
from overrides import overrides

class CEwithMuVarCov(nn.Module):
    def __init__(self,args):
        super(CEwithMuVarCov, self).__init__()
        self.args=args
        self.num_class = 2

    def forward(self, in_feature,label):

        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)

        var_loss = 1/(torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss = - torch.mean(distance) / math.sqrt(in_feature.size(1))

        corr = self.corr(in_feature.T)**2

        loss_corr = torch.mean(corr[torch.tril(torch.ones_like(corr), diagonal=-1) == 1])

        return self.args.lambda_loss_mu * mean_loss + self.args.lambda_loss_var * var_loss+ self.args.lambda_loss_corr*loss_corr, (self.args.lambda_loss_mu * mean_loss, self.args.lambda_loss_var * var_loss, self.args.lambda_loss_corr*loss_corr)



    # def corr(self, X):
    #     D = X.shape[-1]
    #     mean = torch.mean(X, dim=-1).unsqueeze(-1)
    #     X = X - mean
    #     cov = 1 / (D - 1) * X @ X.transpose(-1, -2)
    #
    #     inv = torch.inverse(torch.diag(torch.diag(cov)).sqrt())
    #     corr_mat = inv @ cov @ inv
    #
    #     return corr_mat

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return (1 / (D - 1)) * (X @ X.transpose(-1, -2))

    def corr(self, x):
        # calculate covariance matrix of rows
        c = self.cov(x)

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c) + 1e-8)
        c = c.div(stddev.expand_as(c).t() + 1e-8)

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        # c = torch.clamp(c, -1.0, 1.0)

        return c


# class CEwithMuVarCov2(nn.Module):
#     def __init__(self,args):
#         super(CEwithMuVarCov2, self).__init__()
#         self.args=args
#         self.num_class = 2
#
#     def forward(self, in_feature,label):
#
#         mu_matrix = torch.cat(
#             [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
#             dim=0)
#
#         var_loss = 1/(torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
#         pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
#         distance = pairwise_distance[
#             torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
#         # mean_loss =  -torch.mean(distance) / math.sqrt(in_feature.size(1))
#         mean_loss = math.sqrt(in_feature.size(1))/(torch.mean(distance) + (1e-6))
#
#         corr = self.corr(in_feature.T)**2
#
#         loss_corr = torch.mean(corr[torch.tril(torch.ones_like(corr), diagonal=-1) == 1])
#
#         return self.args.lambda_loss_mu * mean_loss + self.args.lambda_loss_var * var_loss+ self.args.lambda_loss_corr*loss_corr, (self.args.lambda_loss_mu * mean_loss, self.args.lambda_loss_var * var_loss, self.args.lambda_loss_corr*loss_corr)
class CEwithMuVarCov2(nn.Module):
    def __init__(self, args):
        super(CEwithMuVarCov2, self).__init__()
        self.args = args
        self.num_class = 2

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def forward(self, in_feature, label):
        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)

        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss =  -torch.mean(distance) / math.sqrt(in_feature.size(1))
        # mean_loss = math.sqrt(in_feature.size(1)) / (torch.mean(distance) + (1e-6))

        entropy = 0
        for l in range(0, self.num_class):
            cls_data = in_feature[label == l]
            covar = self.cov(cls_data.T) + 5e-4 * torch.eye(cls_data.size(1)).to('cuda')
            entropy += torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(cls_data, dim=0),
                                                                                   covariance_matrix=covar).entropy()
        entropy = entropy / self.num_class
        var_loss = torch.tensor(1.).to('cuda') / (entropy + (1e-6))

        return self.args.lambda_loss_mu * mean_loss + self.args.lambda_loss_var * var_loss, (self.args.lambda_loss_mu * mean_loss, self.args.lambda_loss_var * var_loss, self.args.lambda_loss_corr * torch.zeros(1).to('cuda').long())
    
class CEwithMuVarCov3(nn.Module):
    def __init__(self, args):
        super(CEwithMuVarCov3, self).__init__()
        self.args = args
        self.num_class = 2

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def forward(self, in_feature, label):
        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)

        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss =  -torch.mean(distance) / math.sqrt(in_feature.size(1))
        # mean_loss = math.sqrt(in_feature.size(1)) / (torch.mean(distance) + (1e-6))

        entropy = 0
        for l in range(0, self.num_class):
            cls_data = in_feature[label == l]
            covar = self.cov(cls_data.T) + 5e-4 * torch.eye(cls_data.size(1)).to('cuda')
            entropy += torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(cls_data, dim=0),
                                                                                   covariance_matrix=covar).entropy()
        entropy = entropy / self.num_class
#         var_loss = torch.tensor(1.).to('cuda') / (entropy + (1e-6))
        var_loss = -entropy

        return self.args.lambda_loss_mu * mean_loss + self.args.lambda_loss_var * var_loss, (self.args.lambda_loss_mu * mean_loss, self.args.lambda_loss_var * var_loss, self.args.lambda_loss_corr * torch.zeros(1).to('cuda').long())

# class HLoss_classwise(nn.Module):
#     def __init__(self):
#         super(HLoss_classwise, self).__init__()
#         self.num_class = 2
#
#     def forward(self, in_feature, label):
#         entropy = 0
#         for l in range(0, self.num_class):
#             cls_data = in_feature[label == l]
#             covar = self.cov(cls_data.T) + 5e-4 * torch.eye(cls_data.size(1)).to('cuda')
#             entropy += -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(cls_data, dim=0),
#                                                                                    covariance_matrix=covar).entropy()
#         entropy = entropy / self.num_class
#         loss_entropy = torch.tensor(1.).to('cuda') / (entropy + (1e-6))
#         return loss_entropy



    # def corr(self, X):
    #     D = X.shape[-1]
    #     mean = torch.mean(X, dim=-1).unsqueeze(-1)
    #     X = X - mean
    #     cov = 1 / (D - 1) * X @ X.transpose(-1, -2)
    #
    #     inv = torch.inverse(torch.diag(torch.diag(cov)).sqrt())
    #     corr_mat = inv @ cov @ inv
    #
    #     return corr_mat

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return (1 / (D - 1)) * (X @ X.transpose(-1, -2))

    def corr(self, x):
        # calculate covariance matrix of rows
        c = self.cov(x)

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c) + 1e-8)
        c = c.div(stddev.expand_as(c).t() + 1e-8)

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        # c = torch.clamp(c, -1.0, 1.0)

        return c

# class CEwithMuVarCov2(nn.Module):
#     def __init__(self,args):
#         super(CEwithMuVarCov2, self).__init__()
#         self.args=args
#         self.num_class = 2
#
#     def forward(self, in_feature,label):
#
#         mu_matrix = torch.cat(
#             [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
#             dim=0)
#
#         var_loss = torch.mean(1/(torch.var(in_feature, dim=0) + (1e-6)))
#         pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
#         distance = pairwise_distance[
#             torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
#         mean_loss = - torch.mean(distance)
#
#         corr = self.corr(in_feature.T)**2
#
#         loss_corr = torch.mean(corr[torch.tril(torch.ones_like(corr), diagonal=-1) == 1])
#
#         return self.args.lambda_loss_mu * mean_loss + self.args.lambda_loss_var * var_loss+ self.args.lambda_loss_corr*loss_corr, (self.args.lambda_loss_mu * mean_loss, self.args.lambda_loss_var * var_loss, self.args.lambda_loss_corr*loss_corr)
#
#
#
#     # def corr(self, X):
#     #     D = X.shape[-1]
#     #     mean = torch.mean(X, dim=-1).unsqueeze(-1)
#     #     X = X - mean
#     #     cov = 1 / (D - 1) * X @ X.transpose(-1, -2)
#     #
#     #     inv = torch.inverse(torch.diag(torch.diag(cov)).sqrt())
#     #     corr_mat = inv @ cov @ inv
#     #
#     #     return corr_mat
#
#     def cov(self, X):
#         D = X.shape[-1]
#         mean = torch.mean(X, dim=-1).unsqueeze(-1)
#         X = X - mean
#         return (1 / (D - 1)) * (X @ X.transpose(-1, -2))
#
#     def corr(self, x):
#         # calculate covariance matrix of rows
#         c = self.cov(x)
#
#         # normalize covariance matrix
#         d = torch.diag(c)
#         stddev = torch.pow(d, 0.5)
#         c = c.div(stddev.expand_as(c) + 1e-8)
#         c = c.div(stddev.expand_as(c).t() + 1e-8)
#
#         # clamp between -1 and 1
#         # probably not necessary but numpy does it
#         # c = torch.clamp(c, -1.0, 1.0)
#
#         return c