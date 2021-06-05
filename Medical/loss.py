import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import math
from overrides import overrides
import numpy as np

class BaseLoss(_Loss):  # CrossEntropy Loss
    def __init__(self,args,fc:nn.Linear,num_class=2,hidden_size=768):
        super(BaseLoss, self).__init__()
        self.args=args
        self.fc = fc.to(torch.float32) # (hidden,n_class)

        self.w=self.fc.weight # self.w (n_class,hidden_size)
        self.b=self.fc.bias # n_class
        self.loss_func=nn.CrossEntropyLoss(ignore_index=-1)
        self.num_class=num_class


    def forward(self, in_feature,label):
        """
        in_feature (N, hidden_size)
        """
        in_feature = in_feature.to(torch.float32)
        logits=self.fc(in_feature)

        return self.loss_func(torch.log_softmax(logits,-1),label)


    @torch.no_grad()
    def predict(self, in_feature, mu_matrix=None):
        in_feature = in_feature.to(torch.float32)
        logits = self.fc(in_feature)

        return torch.argmax(logits, dim=-1), torch.softmax(logits, -1),


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

    
class CEwithContrastive(BaseLoss):   # without margin
    def __init__(self, args, fc, margin=0.3, margin_var=None):
        super(CEwithContrastive, self).__init__(args, fc)
        self.margin = margin
        self.zero = torch.FloatTensor([0.0]).to(self.fc.weight.device)
        self.margin_var = margin_var
        self.basic_loss = BaseLoss(args, fc)
       

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    @overrides
    def forward(self, in_feature, label):
        """
        in_feature : (N, hidden_size) torch.float32
        label : (N) torch.int64
        """

        n_sample = in_feature.shape[0]
        trg = label[:, None]

        mu_matrix = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
                         dim=0)
        l2_distance = torch.cdist(in_feature, mu_matrix)

        basic_loss = self.basic_loss(in_feature, label)
        var_loss = 0.0
        mean_loss = 0.0
        var_loss_new = torch.mean(1 / (torch.var(in_feature, dim=0) + (1e-6)))
        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss = - torch.mean(distance)
        
        return basic_loss, mean_loss, var_loss_new

    @overrides
    def predict(self,in_feature):
        """
        mu_matrix : (n_class, hidden)
        """
        in_feature = in_feature.to(torch.float32)
        logits = self.fc(in_feature)
        values, incides = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        return torch.argmax(logits,dim=-1),  values


class CEwithMuVarCorr(BaseLoss):
    def __init__(self, args, fc, margin = 0.01, margin_var = 0.01, margin_corr = 0.01):
        super(CEwithMuVarCorr, self).__init__(args, fc)
        self.args = args
        self.fc = fc
        self.margin = margin
        self.margin_var = margin_var
        self.margin_corr = margin_corr
        self.basic_loss =  BaseLoss(args, fc)

    @overrides
    def forward(self, in_feature, label):
        hidden_size = in_feature.size(-1)
        mu_matrix = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
                         dim=0)
        basic_loss = self.basic_loss(in_feature, label)
        var_loss = torch.tensor(1.).to('cuda') / (torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        mean_loss = - torch.mean(distance) / math.sqrt(hidden_size)

        corr = self.corr(in_feature.T) ** 2
        loss_corr = torch.mean(torch.tril(corr, -1))

        return basic_loss + self.margin * mean_loss + self.margin_var * var_loss + self.margin_corr * loss_corr

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

        return c

    @overrides
    def predict(self,in_feature):
        """
        mu_matrix : (n_class, hidden)
        """
        in_feature = in_feature.to(torch.float32)
        logits = self.fc(in_feature)
        values, incides = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        return torch.argmax(logits,dim=-1),  values