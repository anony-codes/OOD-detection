import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn

import math
import numpy as np
from overrides import overrides


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x



class BaseLoss(_Loss):  # inner product
    def __init__(self,args,fc:nn.Linear,num_class=2):
        super(BaseLoss, self).__init__()
        self.args=args
        self.fc=fc # (hidden,n_class)
        self.w=self.fc.weight # self.w (n_class,hidden_size)
        self.b=self.fc.bias # n_class
        self.loss_func=nn.CrossEntropyLoss(ignore_index=-1)
        self.num_class=num_class

    def forward(self, in_feature,label):
        """
        in_feature (N, hidden_size)
        """
        # # if self.args.set_bias:
        # #     logits=torch.mm(in_feature.to(self.w.dtype),self.w.T)+self.b
        # # else:
        # logits = torch.mm(in_feature.to(self.w.dtype), self.w.T)
        logits=self.fc(in_feature.to(self.fc.weight.dtype))

        #to visualize weight distance on tensorboard
        pos = label
        neg = 1-label # only apply in binary case

        pos_feature = torch.index_select(self.w, 0, index=pos)
        neg_feature = torch.index_select(self.w, 0, index=neg)

        return self.loss_func(logits, label)
        # return self.loss_func(torch.log_softmax(logits,-1),label) #,pos_mean.item(),neg_mean.item()

    def predict(self,in_feature):

        logits=self.fc(in_feature.to(self.fc.weight.dtype))
        # values, incides = torch.max(torch.softmax(logits,dim=-1), dim=-1)

        # return torch.argmax(logits, dim=-1), values
        return torch.argmax(logits,dim=-1),torch.softmax(logits,dim=-1)

    def predict_cos_similarity(self,in_feature, mu_matrix = None, flag_mu = False):
        in_feature=in_feature.to(self.fc.weight.dtype)
        f = F.normalize(in_feature, p=2, dim=-1)
        w = F.normalize(self.fc.weight,p=2,dim=-1)
        if flag_mu:
            w = F.normalize(mu_matrix, p=2, dim=-1)
        wf = torch.matmul(f,w.T)

        values, incides = torch.max(wf, dim=-1)
        return torch.argmax(wf, dim=-1), values


class ContrastiveLoss(BaseLoss):
    def __init__(self,args,fc,margin=64):
        super(ContrastiveLoss, self).__init__(args,fc)
        self.margin=margin
        self.zero=torch.FloatTensor([0.0]).to(self.fc.weight.device)

    @overrides
    def forward(self, in_feature,label):
        """\
        in_feature : (N, hidden_size) torch.float32
        label : (N) torch.int64
        """

        n_sample=in_feature.shape[0]
        trg = label[:, None]

        if self.args.learn_type=="mu":
            mu_matrix = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)], dim=0)
            l2_distance = torch.cdist(in_feature,mu_matrix)

        elif self.args.learn_type=="w":

            l2_distance = torch.cdist(in_feature,self.fc.weight.to(in_feature.dtype))

        else:
            raise NotImplementedError

        one_hot_target = (trg == torch.arange(self.num_class).to(l2_distance.device).reshape(1, self.num_class)).bool() # all false distance index
        term1 = l2_distance.gather(dim=1,index=label.unsqueeze(1)) # true distance
        term2 = l2_distance[~one_hot_target].view([n_sample,-1]) # False distance

        if self.args.loss=="contrastive":
            pos_mean = torch.mean(term1)
            neg_mean = torch.mean(term2)

            term2 = self.margin - term2
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            # term2 = torch.mean(term2,dim=-1) # discuss more
            # term1 = torch.mean(term1,dim=-1) # discuss more
            loss = term1 + term2

            return torch.mean(loss), pos_mean, neg_mean

        if self.args.loss == "triplet":
            pos_mean = torch.mean(term1)
            neg_mean = torch.mean(term2)

            loss_1=term1 - term2 + self.margin
            loss=torch.where(loss_1>=self.zero,loss_1,self.zero)

            return torch.mean(loss), pos_mean, neg_mean

        if self.args.loss == "contrastive_mu_v2": #  version for  preserving std by 1
            pos_mean = torch.mean(term1)
            neg_mean = torch.mean(term2)

            term2 = self.margin - term2
            term1 = term1-1

            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            term1 = torch.where(term1 >= self.zero, term1, self.zero)

            loss=term1+term2

            return torch.mean(loss) #, pos_mean.item(), neg_mean.item()


    @overrides
    def predict(self,in_feature,mu_matrix=None, pred_type = "dist"):
        """
        in_feature : (N, hidden_size) torch.float32
        mu_matrix : (n_class, hidden_size ) torch.float32
        """

        if self.args.learn_type == "mu":
            dist = torch.cdist(in_feature,mu_matrix)
        elif self.args.learn_type == "w":
            dist = torch.cdist(in_feature, self.fc.weight.to(in_feature.dtype))
        else:
            raise NotImplementedError

        if self.args.pred_type == 'dist':
            values, indices = torch.max(-dist, dim=-1)

        elif self.args.pred_type == 'normalized':
            inv_dist= 1 / (dist + 1e-5)
            normalized_prob = inv_dist / torch.sum(inv_dist, dim=-1).unsqueeze(-1)

            values, indices = torch.max(normalized_prob, dim=-1)
        elif self.args.pred_type == "softmax":
            values, incides = torch.max(torch.softmax(-dist, dim=-1), dim=-1)
        else:
            raise NotImplementedError
        return torch.argmin(dist,dim=-1), values #normalized_prob


class CEwithContrastive(BaseLoss):
    def __init__(self, args, fc, margin=0.3, margin_var=None):
        super(CEwithContrastive, self).__init__(args, fc)
        self.margin = margin
        self.zero = torch.FloatTensor([0.0]).to(self.fc.weight.device)
        self.basic_loss = BaseLoss(args, fc)
        if margin_var is not None:
            self.margin_var = margin_var

    """
    ref: https://github.com/pytorch/pytorch/issues/19037
    """
    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    @overrides
    def forward(self, in_feature, label):
        """\
        in_feature : (N, hidden_size) torch.float32
        label : (N) torch.int64
        """

        n_sample=in_feature.shape[0]
        trg = label[:, None]

        if self.args.learn_type=="mu":
            flag = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)], dim=0)
            l2_distance = torch.cdist(in_feature,flag)
            # l2_distance = in_feature[:, None] - flag[None]

        elif self.args.learn_type=="w":
            flag = self.fc.weight.to(in_feature.dtype)
            l2_distance = torch.cdist(in_feature,flag)
            # l2_distance = in_feature[:, None] - flag[None]

        else:
            raise NotImplementedError

        basic_loss= self.basic_loss(in_feature, label)

        if self.args.loss=="ce_with_contrastive":
            # l2_distance = torch.sqrt(torch.diagonal(torch.matmul(term, term.transpose(1, 2)), dim1=1, dim2=2))
            one_hot_target = (trg == torch.arange(self.num_class).to(l2_distance.device).reshape(1, self.num_class)).bool()  # all false distance index
            term2 = l2_distance[~one_hot_target].view([n_sample, -1])
            term = self.margin - term2
            term = torch.where(term >= self.zero, term, self.zero)

            return basic_loss + torch.mean(term), basic_loss, torch.mean(term)

        # elif self.args.loss=="ce_with_triplet(full)":
        #     one_hot_target = (trg == torch.arange(self.num_class).to(l2_distance.device).reshape(1, self.num_class)).bool()  # all false distance index
        #     term1 = l2_distance.gather(dim=1, index=label.unsqueeze(1))  # true distance
        #     term2 = l2_distance[~one_hot_target].view([n_sample, -1])  # False distance
        #
        #     loss_1 = term1 - term2 + self.margin
        #     loss = torch.where(loss_1 >= self.zero, loss_1, self.zero)
        #
        #     return basic_loss + torch.mean(loss), basic_loss, torch.mean(loss)
        #
        # elif self.args.loss=="ce_with_contrastive(full)":
        #     one_hot_target = (trg == torch.arange(self.num_class).to(l2_distance.device).reshape(1, self.num_class)).bool()  # all false distance index
        #     term1 = l2_distance.gather(dim=1, index=label.unsqueeze(1))  # true distance
        #     term2 = l2_distance[~one_hot_target].view([n_sample, -1])  # False distance
        #
        #     term2 = self.margin - term2
        #     term2 = torch.where(term2 >= self.zero, term2, self.zero)
        #
        #     loss = term1 + term2
        #     return basic_loss + torch.mean(loss), basic_loss, torch.mean(loss)

        elif self.args.loss=="ce_with_mu": # margin with

            # pairwise_extracted = flag[:,None] - flag[None]
            pairwise_distance=torch.cdist(flag,flag,p=2)
            # distance = torch.matmul(pairwise_extracted, pairwise_extracted.transpose(1, 2))
            # distance = torch.sqrt(torch.tril(torch.diagonal(distance, dim1=1, dim2=2)))
            distance = pairwise_distance[torch.tril(torch.ones_like(pairwise_distance),diagonal=-1)==1] # pick lower triangular
            term2 = self.margin - distance
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            # term2 = torch.exp( -1*distance)

            return basic_loss + torch.mean(term2), basic_loss, torch.mean(term2)   # , pos_mean.item(), neg_mean.item()

        elif self.args.loss=="ce_with_exponential": # no margin
            # l2_distance = torch.sqrt(torch.diagonal(torch.matmul(term, term.transpose(1, 2)), dim1=1, dim2=2))
            one_hot_target = (trg == torch.arange(self.num_class).to(l2_distance.device).reshape(1, self.num_class)).bool()  # all false distance index
            term2 = l2_distance[~one_hot_target].view([n_sample, -1])
            term2 = torch.exp( -1*term2)

            return basic_loss + torch.mean(term2), basic_loss, torch.mean(term2)   # , pos_mean.item(), neg_mean.item()

        elif (self.args.loss=="ce_with_mu_var")|(self.args.loss == 'ce_with_mu_var_ver2'):
            pairwise_distance=torch.cdist(flag,flag,p=2)
            distance = pairwise_distance[torch.tril(torch.ones_like(pairwise_distance),diagonal=-1)==1] # pick lower triangular
            term2 = self.margin - distance
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            loss_mu = torch.mean(term2)

            covs = [self.cov(in_feature[label==l].T)[None, ...] for l in range(0, self.num_class)]
            covs = torch.cat(covs, dim=0)

            if self.args.loss=="ce_with_mu_var":
                trgs = [self.margin_var*torch.eye(cov.shape[0])[None, ...].to(covs.device) for cov in covs]
                trgs = torch.cat(trgs, dim=0)
                # loss_var = nn.L1Loss(covs, trgs)
                loss_var = nn.MSELoss()(covs, trgs)

            elif self.args.loss == 'ce_with_mu_var_ver2':
                covs = abs(covs)
                trgs = self.margin_var * torch.ones_like(covs).to(covs.device)
                diff = covs - trgs
                loss_var = torch.where(diff >= self.zero, diff, self.zero)
                loss_var = torch.mean(loss_var)
            else:
                raise NotImplementedError

            return basic_loss + loss_mu + loss_var, basic_loss, (loss_mu, loss_var)

        elif (self.args.loss=="mu_var") | (self.args.loss=="mu_var_ver2"):
            pairwise_distance=torch.cdist(flag,flag,p=2)
            distance = pairwise_distance[torch.tril(torch.ones_like(pairwise_distance),diagonal=-1)==1] # pick lower triangular
            term2 = self.margin - distance
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            loss_mu = torch.mean(term2)

            covs = [self.cov(in_feature[label==l].T)[None, ...] for l in range(0, self.num_class)]
            covs = torch.cat(covs, dim=0)

            if self.args.loss=="mu_var":
                trgs = [self.margin_var*torch.eye(cov.shape[0])[None, ...].to(covs.device) for cov in covs]
                trgs = torch.cat(trgs, dim=0)
                # loss_var = nn.L1Loss(covs, trgs)
                loss_var = nn.MSELoss()(covs, trgs)

            elif self.args.loss == 'mu_var_ver2':
                covs = abs(covs)
                trgs = self.margin_var * torch.ones_like(covs).to(covs.device)
                diff = covs - trgs
                loss_var = torch.where(diff >= self.zero, diff, self.zero)
                loss_var = torch.mean(loss_var)

            else:
                raise NotImplementedError

            return loss_mu + loss_var, loss_mu, loss_var

        elif (self.args.loss == "ce_with_mu_var_diag") | (self.args.loss == 'mu_var_diag'):
            pairwise_distance = torch.cdist(flag, flag, p=2)
            distance = pairwise_distance[
                torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
            term2 = self.margin - distance
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
            loss_mu = torch.mean(term2)

            covs = [torch.diagonal(self.cov(in_feature[label == l].T))[None, ...] for l in range(0, self.num_class)]
            covs = torch.cat(covs, dim=0)

            trgs = self.margin_var * torch.ones_like(covs).to(covs.device)
            diff = covs - trgs
            loss_var = torch.where(diff >= self.zero, diff, self.zero)
            loss_var = torch.mean(loss_var)

            if self.args.loss == "ce_with_mu_var_diag":

                return basic_loss + loss_mu + 100*loss_var, basic_loss, (loss_mu, loss_var)

            elif self.args.loss == 'mu_var_diag':
                return loss_mu + loss_var, loss_mu, loss_var


            else:
                raise NotImplementedError



        else:
            raise NotImplementedError



    @overrides
    def predict(self,in_feature,mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        logits = self.fc(in_feature.to(self.fc.weight.dtype))
        values, incides = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        # return torch.argmax(logits,dim=-1),torch.softmax(logits,dim=-1)
        return torch.argmax(logits, dim=-1), values

    def predict_dist(self,in_feature,mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        dist = torch.cdist(in_feature,mu_matrix)
        values, indices = torch.max(-dist, dim=-1)

        return torch.argmin(dist, dim=-1), values



class OVADMLoss(BaseLoss):
    def __init__(self,args,fc):
        super(OVADMLoss, self).__init__(args,fc)
        """
        self.fc.weight : (n_class,hidden_size)
        """

        self.num_class=self.fc.weight.shape[0]

    @overrides
    def forward(self, in_feature,label):
        """\
        in_feature : (N, hidden_size) torch.float32
        label : (N) torch.int64
        """

        if self.args.learn_type=="mu":
            mu_matrix = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)], dim=0)
            l2_distance = torch.cdist(in_feature,mu_matrix)

        elif self.args.learn_type=="w":
            # weight=self.fc.weight.to(in_feature.dtype)
            # term = in_feature[: , None] - self.fc.weight[None]
            l2_distance = torch.cdist(in_feature,self.fc.weight.to(in_feature.dtype))
            # l2_distance = torch.cdist(in_feature,self.fc.weight)
            # l2_distance = torch.sqrt(torch.diagonal(torch.matmul(term, term.transpose(1, 2)), dim1=1, dim2=2))
        else:
            raise NotImplementedError

        output = 2 * nn.Sigmoid()(l2_distance * -1)
        loss=self._ova(output,label)

        return loss

    def _ova(self, prob, label):
        loss = -torch.log(prob.gather(1, label.unsqueeze(1)) + 1e-8)
        trg = label[:, None]
        one_hot_target = (trg == torch.arange(self.num_class).to(prob.device).reshape(1, self.num_class)).bool()
        reg_loss = torch.log(1 - prob + 1e-8) * (~one_hot_target)
        reg_loss = reg_loss.sum()
        final_loss = torch.sum(loss) - reg_loss

        return final_loss / float(len(prob))

    @overrides
    def predict(self,in_feature, mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        if self.args.learn_type=="mu":
            # term = in_feature[:, None] - mu_matrix[None]
            l2_distance = torch.cdist(in_feature,mu_matrix)

        elif self.args.learn_type == "w":
            l2_distance = torch.cdist(in_feature,self.fc.weight.to(in_feature.dtype))
            # l2_distance = torch.cdist(in_feature,self.fc.weight)

        output = 2 * nn.Sigmoid()(l2_distance * -1)
        # return torch.argmax(output, dim=-1), output
        values, incides = torch.max(output, dim=-1)
        return torch.argmax(output,dim=-1), values

class AngularPenaltySMLoss(BaseLoss):

    def __init__(self,args,fc, loss_type='arcface', eps=1e-6, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__(args,fc)
        loss_type = loss_type.lower()

        assert loss_type in ['arcface', 'sphereface', 'cosface']

        if loss_type == 'arcface':
            self.s = 4.0 if not s else s
            self.m = 0.75 if not m else m

        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m

        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m

        self.loss_type = loss_type
        self.eps = eps

    @overrides
    def forward(self, in_feature, label):
        '''
        input shape (N, in_features)
        labels ( N )
        '''
        assert len(in_feature) == len(labels)
        assert torch.min(labels) >= 0
        # in_feature=in_feature
        # in_feature = self.w.to(in_feature.dtype)
        # assert torch.max(labels) < self.out_features
        w = norm(self.w)
        f = norm(in_feature)

        if self.args.learn_type == 'mu':
            mu_matrix = torch.cat([torch.mean(in_feature[labels == l], dim=0).unsqueeze(0) for l in range(0, self.args.num_classes)], dim=0)
            w = norm(mu_matrix)

        wf = torch.matmul(f,w.T)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)

        # dist_mu, dist_fc, std_mu, mu_means, w_means = self.calc_dist(in_feature=in_feature,label=labels,fc=self.fc,args=self.args)
        # for_visualize={"dist_mu":dist_mu ,"dist_fc":dist_fc ,"std_mu":std_mu ,"mu_means":mu_means ,"w_means":w_means}

        return -torch.mean(L)#,for_visualize


    @overrides
    def predict(self,in_feature, mu_matrix=None, flag_mu = False):
        in_feature=in_feature.to(self.w.dtype)

        w = F.normalize(self.w,p=2,dim=-1)
        if (self.args.learn_type == 'mu') | (flag_mu):
            w = F.normalize(mu_matrix, p=2, dim=-1)
        f = F.normalize(in_feature,p=2,dim=-1)
        wf = self.s*torch.matmul(f,w.T)

        values, incides = torch.max(wf, dim=-1)
        return torch.argmax(wf, dim=-1), values

        # return torch.argmax(wf,dim=-1), wf

    def predict_cos_test(self,in_feature, mu_matrix=None, flag_mu = False):
        in_feature=in_feature.to(self.w.dtype)

        w = F.normalize(self.w,p=2,dim=-1)
        if (self.args.learn_type == 'mu') | (flag_mu):
            w = F.normalize(mu_matrix, p=2, dim=-1)
        f = F.normalize(in_feature,p=2,dim=-1)
        # wf = self.s*torch.matmul(f,w.T)
        wf = torch.matmul(f, w.T)

        values, incides = torch.max(wf, dim=-1)
        return torch.argmax(wf, dim=-1), values

    def predict_softmax(self,in_feature, mu_matrix=None, flag_mu = False):
        in_feature=in_feature.to(self.w.dtype)

        w = F.normalize(self.w,p=2,dim=-1)
        if (self.args.learn_type == 'mu') | (flag_mu):
            w = F.normalize(mu_matrix, p=2, dim=-1)
        f = F.normalize(in_feature,p=2,dim=-1)
        # wf = self.s*torch.matmul(f,w.T)
        wf = torch.matmul(f, w.T)

        values, incides = torch.max(torch.softmax(wf, dim=-1), dim=-1)
        return torch.argmax(wf, dim=-1), values


class NTXentLoss(torch.nn.Module):
    def __init__(self, args, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        self.batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        #self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        if 'ce_simclr_negative' in self.args.loss:
            positives = torch.ones_like(positives)

        negatives = similarity_matrix[self.mask_samples_from_same_repr[:(self.batch_size*2), :(self.batch_size*2)]].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class SVD_L(torch.nn.Module):
    def __init__(self, args):
        super(SVD_L, self).__init__()
        self.args = args

    def forward(self, in_feature, labels):
        in_feature = in_feature.to(torch.float32)
        loss = torch.tensor(1.).to('cuda')/(torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        loss *= self.args.lambda_loss_var
        return loss



class Uniform_L(torch.nn.Module):
    def __init__(self, args):
        super(Uniform_L, self).__init__()
        self.args = args

    def forward(self, in_feature, label):
        in_feature = in_feature.to(torch.float32)
        sq_pdist = torch.pdist(in_feature, p=2).pow(2)
        loss = sq_pdist.mul(-self.args.temperature).exp().mean().add(1e-6).log()
        loss *= self.args.lambda_loss_var
        return loss

class CEwithMu(BaseLoss):
    def __init__(self, args, fc, margin=0.3, margin_var=None):
        super(CEwithMu, self).__init__(args, fc)
        self.margin = margin
        self.zero = torch.FloatTensor([0.0]).to(self.fc.weight.device)
        self.args = args
        self.basic_loss = BaseLoss(args, fc)
        if margin_var is not None:
            self.margin_var = margin_var

    """
    ref: https://github.com/pytorch/pytorch/issues/19037
    """
    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    @overrides
    def forward(self, in_feature, label):
        in_feature = in_feature.to(torch.float32)

        class_mu = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)
        pairwise_distance = torch.cdist(class_mu, class_mu, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        if self.args.flag_margin:
            term2 = self.args.M - distance
            term2 = torch.where(term2 >= self.zero, term2, self.zero)
        else:
            term2 = -distance
            term2 *= self.args.lambda_loss_mu
        loss_mu = torch.mean(term2)

        # if self.args.loss == 'ce_with_mu_svd':
        #     loss_mu = 0.001*loss_mu

        basic_loss = self.basic_loss(in_feature, label)

        tot_loss = basic_loss + loss_mu

        return tot_loss, basic_loss, loss_mu

    # def forward(self, in_feature, label):
    #     """\
    #     in_feature : (N, hidden_size) torch.float32
    #     label : (N) torch.int64
    #     """
    #
    #     n_sample=in_feature.shape[0]
    #     trg = label[:, None]
    #
    #     if self.args.learn_type=="mu":
    #         flag = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)], dim=0)
    #         l2_distance = torch.cdist(in_feature,flag)
    #     else:
    #         raise NotImplementedError
    #
    #     basic_loss= self.basic_loss(in_feature, label)
    #
    #     if self.args.loss=="ce_with_mu": # margin with
    #         pairwise_distance=torch.cdist(flag,flag,p=2)
    #         distance = pairwise_distance[torch.tril(torch.ones_like(pairwise_distance),diagonal=-1)==1] # pick lower triangular
    #         term2 = self.margin - distance
    #         term2 = torch.where(term2 >= self.zero, term2, self.zero)
    #
    #         return basic_loss + torch.mean(term2), basic_loss, torch.mean(term2)   # , pos_mean.item(), neg_mean.item()
    #
    #     else:
    #         raise NotImplementedError

    @overrides
    def predict(self,in_feature,mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        logits = self.fc(in_feature.to(self.fc.weight.dtype))
        values, incides = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        # return torch.argmax(logits,dim=-1),torch.softmax(logits,dim=-1)
        return torch.argmax(logits, dim=-1), values

    def predict_dist(self,in_feature,mu_matrix=None):
        """
        mu_matrix : (n_class, hidden)
        """
        dist = torch.cdist(in_feature,mu_matrix)
        values, indices = torch.max(-dist, dim=-1)

        return torch.argmin(dist, dim=-1), values

class LabelSmoothingCrossEntropy(BaseLoss):
    def __init__(self, args, fc):
        super(LabelSmoothingCrossEntropy, self).__init__(args,fc)
        self.smoothing = args.smooth_factor
        self.args = args

    def forward(self, in_feature, target):

        in_feature = in_feature.to(self.fc.weight.dtype)
        x = self.fc(in_feature)

        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
    
# class HLoss(nn.Module):
#     """
#     https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510/2
#     """
#     def __init__(self):
#         super(HLoss, self).__init__()
#
#     def forward(self, x):
#         loss = torch.sum(torch.stack([-torch.distributions.normal.Normal(mu, std).entropy() for mu, std in
#                                       zip(torch.mean(out, dim=0), torch.std(out, dim=0))]))
#         x = x.T
#         b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
#         b = -1.0 * b.sum()
#
#         b = 0.01 * b
#         return b


# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()
#
#     def forward(self, x):
#         loss = torch.mean(-self.entropy(torch.std(x, dim=0)))
#
#         return loss
#
#     def entropy(self, scale):
#         return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale + 1e-8)



class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        cov = self.cov(x.T) + 1e-6*torch.eye(x.size(1)).to('cuda')
        # cov = self.cov(x.T)
        # loss = -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim=0), covariance_matrix=cov).entropy()
#         loss = -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim=0),
#                                                                            scale_tril=torch.tril(cov)).entropy()
        entropy = torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim=0),
                                                                           scale_tril=torch.tril(cov)).entropy()
        
        loss = torch.tensor(1.).to('cuda')/(entropy + (1e-6))

        return loss

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

class HLoss_classwise(nn.Module):
    def __init__(self):
        super(HLoss_classwise, self).__init__()
        self.num_class = 2

    def forward(self, in_feature, label):
        entropy = 0
        for l in range(0, self.num_class):
            cls_data = in_feature[label == l]
            covar = self.cov(cls_data.T) + 5e-4 * torch.eye(cls_data.size(1)).to('cuda')
            entropy += -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(cls_data, dim=0),
                                                                           covariance_matrix=covar).entropy()
        entropy = entropy/self.num_class
        loss_entropy = torch.tensor(1.).to('cuda') / (entropy + (1e-6))
        return loss_entropy

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)
