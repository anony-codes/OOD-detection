from .losses import BaseLoss
import math
import torch
import numpy as np
import torch.nn as nn
from overrides import overrides


class NTXentLoss(BaseLoss):
    def __init__(self, args, fc, num_class, batch_size, temperature=1.0, use_cosine_similarity=True, hidden_size=512,
                 device="cuda"):
        # def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__(args, fc, num_class)
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.pooler_simclr = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.__init_classifier_weight()

        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion_simclr = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion_ce = torch.nn.CrossEntropyLoss()

    def __init_classifier_weight(self):
        nn.init.xavier_normal(self.pooler_simclr.weight.data)

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

    def _simclr(self, hi, hj):

        zis = self.pooler_simclr(hi)
        zjs = self.pooler_simclr(hj)

        self.batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[
            self.mask_samples_from_same_repr[:(self.batch_size * 2), :(self.batch_size * 2)]].view(2 * self.batch_size,
                                                                                                   -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion_simclr(logits, labels)

        return loss / (2 * self.batch_size)

    def forward(self, in_features, label):

        in_features = [in_feature.to(torch.float32) for in_feature in in_features]
        # in_feature = in_feature.to(torch.float32)

        fi = in_features[0]
        fj = in_features[1]

        logits_1 = self.fc(fi)
        logits_2 = self.fc(fj)

        loss_simclr = self._simclr(fi, fj)
        loss_ce = (self.criterion_ce(logits_1, label) + self.criterion_ce(logits_2, label)) / 2

        return loss_ce + loss_simclr