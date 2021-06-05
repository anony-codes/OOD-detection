from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
import apex
import random
import math
import pandas as pd
import os
import numpy as np
from overrides import overrides
from sklearn.metrics import classification_report
from .mahalanobis import get_Mahalanobis_score_penulti,get_Mahalanobis_score
from model.losses import *
from util.odin import get_odin_score
from torch.utils.tensorboard import SummaryWriter
from .logger import Logger
import shutil


class Trainer:
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria: BaseLoss, clip_norm, mixed_precision, num_class: int = 2):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm
        self.num_class = num_class

        self.__init_writer()

    def __init_writer(self):
        board_directory = os.path.join(self.args.vis_dir, self.args.savename)
        if not os.path.isdir(board_directory):
            os.makedirs(board_directory)
        else:
            shutil.rmtree(board_directory)
            os.makedirs(board_directory)

        self.writer = Logger(log_dir=board_directory)

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True, drop_last=True)

        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=len(batchfier.dataset))

        for inp in pbar:
            inp = self.reformat_inp(inp)
            # print(inp[0].shape)
            # print(inp[-1].shape)
            in_feature, _, _ = model(inp[0])
            # print(in_feature.shape)


            loss = criteria(in_feature, inp[-1].view(-1))
            # self.writer.scalar_summary("loss/train_loss", loss, self.step)

            # for key, value in for_visualize.items():
            #     if "dist" in key:
            #         self.writer.scalar_summary("dist/{}".format(key), value, self.step)
            #     elif "loss" in key:
            #         self.writer.scalar_summary("loss/{}".format(key), value, self.step)
            #     else:
            #         self.writer.scalar_summary("{}".format(key), value, self.step)

            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt), n_bar), )
                pbar.update()

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        model.eval()
        criteria.rez()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        n_samples = 0
        for inp in pbar:
            with torch.no_grad():
                # if 0 in inp[-2]:
                #     continue

                inp = self.reformat_inp(inp)
                logits, _ = model(inp[0])
                loss, _ = criteria(logits, inp[-1])
                step_loss += loss.item()
                pbar_cnt += 1
                description = criteria.get_description(pbar_cnt)
                pbar.set_description(description)
        pbar.close()
        return math.exp(step_loss / pbar_cnt)


class CFTrainer(Trainer):
    def __init__(self, args, model, train_batchfier, dev_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        super(CFTrainer, self).__init__(args, model, train_batchfier, dev_batchfier, optimizers,
                                        update_step, criteria, clip_norm, mixed_precision)

        self.n_label = n_label
        self.mu_matrix = None

    @overrides
    def reformat_inp(self, inp):
        label = inp[-1].to("cuda")

        return inp[0].to("cuda"), label

    def generate_train_statistics(self):

        model = self.model
        batchfier = self.train_batchfier

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        pbar = tqdm(batchfier, total=len(batchfier.dataset))
        model.eval()
        model.zero_grad()

        in_feature_tensors = None
        label_tensors = None

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)

                in_feature, _, _ = model(inp[0], inp[-1])
                if label_tensors is None:
                    label_tensors = gt
                else:
                    label_tensors = torch.cat([label_tensors, gt], dim=0)

                if in_feature_tensors is None:
                    in_feature_tensors = in_feature
                else:
                    in_feature_tensors = torch.cat([in_feature_tensors, in_feature], dim=0)

        mu_matrix = torch.cat(
            [torch.mean(in_feature_tensors[label_tensors == i], 0).unsqueeze(0) for i in range(0, self.num_class)],
            dim=0)
        self.mu_matrix = mu_matrix
        pbar.close()

        return self.mu_matrix

    def test_epoch(self):

        model = self.model
        batchfier = self.test_batchfier

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()

        pbar = tqdm(batchfier, total=len(batchfier.dataset))
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inps in pbar:

            with torch.no_grad():
                inps = self.reformat_inp(inps)
                gt = inps[-1].view(-1)
                in_feature, _, _ = model(inps[0])
                preds, _ = criteria.predict(in_feature)
                loss = criteria(in_feature, gt.view(-1))

                step_loss += loss.item()
                pbar_cnt += 1

                true_buff.extend(gt.tolist())
                eval_buff.extend(preds.tolist())

                score = torch.mean((preds == gt).to(torch.float))
                tot_score += score

                pbar.set_description(
                    "test loss : %f  test accuracy : %f" % (
                        step_loss / pbar_cnt, tot_score / pbar_cnt), )
                pbar.update()

        pbar.close()
        report = classification_report(true_buff, eval_buff, labels=list(range(0, self.n_label)), output_dict=True)
        if "accuracy" in report:
            accuarcy, macro_f1 = report["accuracy"], report["macro avg"]["f1-score"]
        else:
            accuarcy, macro_f1 = report["micro avg"]["f1-score"], report["macro avg"]["f1-score"]

        print("test accuracy: {0:.4f}  test macro_f1: {1:.4f}".format(accuarcy, macro_f1))

        return accuarcy, macro_f1

    def ood_epoch(self, batchfier, mean, prec, var, epsilon, n_class=2):
        model = self.model
        # batchfier = self.test_batchfier

        if self.args.learn_type == "mu":
            if self.mu_matrix is None:
                self.mu_matrix = mean[0]

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        odin_criteria = nn.CrossEntropyLoss()

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)
        model.zero_grad()
        pbar = tqdm(batchfier, total=len(batchfier.dataset))
        pbar_cnt = 0
        tot_score = 0.0

        # if not isinstance(prec,torch.Tensor):
        #     prec=torch.Tensor(prec)

        identity = torch.eye(768).to("cuda")

        ood_for_save = {"text": [], "gt": [], "pred": [], "argmax_prob": [], "prob_dist": [], "m_distance": [],
                        "fx": [], "odin_score": [], "m_distance_pen": []}

        for inp in pbar:
            with torch.no_grad():
                inp = self.reformat_inp(inp)
                gt = inp[-1].view(-1)

                in_feature, fx, penultimate_layer = model(inp[0])

                if self.args.learn_type == "w":
                    preds, probs = criteria.predict(in_feature)
                else:
                    if "mu_var" in self.args.metric:
                        preds, probs = criteria.predict_dist(in_feature, self.mu_matrix)
                    else:
                        preds, probs = criteria.predict(in_feature, self.mu_matrix)

                max_probs, _ = torch.topk(probs, k=1, dim=-1)
                ood_for_save["prob_dist"].extend(probs.tolist())
                ood_for_save["argmax_prob"].extend(max_probs.tolist())

                pbar_cnt += 1

            ood_for_save["text"].extend(inp[0].tolist())
            ood_for_save["gt"].extend(inp[-1].view(-1).tolist())

            ood_for_save["pred"].extend(preds.tolist())

            ood_for_save["fx"].extend(penultimate_layer.tolist())

            ood_outputs = get_Mahalanobis_score(fx, n_class, sample_mean=mean, precision=prec, args=self.args)
            pen_outputs = get_Mahalanobis_score_penulti(fx, n_class, sample_mean=mean, precision=prec, args=self.args)
            odin_outputs = get_odin_score(self.args, model, inp, temperature=1000, epsilon=epsilon,
                                          criterion=odin_criteria,
                                          flag_discrete=True, device="cuda", )

            # print(ood_outputs)
            m_distance, _ = torch.max(ood_outputs.data, 1)

            ood_for_save["m_distance"].extend(m_distance.tolist())
            ood_for_save["m_distance_pen"].extend(pen_outputs.tolist())
            ood_for_save["odin_score"].extend(odin_outputs.tolist())

            score = torch.mean((preds == gt).to(torch.float))
            tot_score += score

            pbar.set_description(
                "ood accuracy : %f" % (tot_score / pbar_cnt), )
            pbar.update()

        return pd.DataFrame(ood_for_save)

import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def knn(model, device, val_loader):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for inp, label in tqdm(val_loader):
            inp,label=reformat_inp((inp,label))
            in_feature,_,_=model(inp)

            output = F.normalize(in_feature, dim=-1).data.cpu()
            features.append(output)
            labels.append(label)



        features = torch.cat(features).numpy()
        labels = torch.cat(labels,dim=0).cpu().numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


def reformat_inp(inp):
    label = inp[-1].to("cuda")

    return inp[0].to("cuda"),label





from util.mahalanobis import get_scores_multi_cluster,get_features

class ContrastiveTrainer(CFTrainer):
    def __init__(self, args, model, train_batchfier, dev_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label):
        CFTrainer.__init__(self, args, model, train_batchfier, dev_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision,n_label)

    @overrides
    def reformat_inp(self, inp):
        inp_tensor =tuple(i.to("cuda") for i in inp[0])
        label = inp[-1].to("cuda")

        return inp_tensor,label


    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True, drop_last=True)

        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=len(batchfier.dataset))

        for inp in pbar:
            inps,label = self.reformat_inp(inp)
            hi, _, _ = model(inps[0])
            hj, _, _ = model(inps[1])

            in_features=torch.cat([hi.unsqueeze(1),hj.unsqueeze(1)],dim=1)

            loss = criteria(in_features, label)

            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                # scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt), n_bar), )
                pbar.update()

        pbar.close()


    def test_epoch(self):

        model = self.model
        batchfier = self.test_batchfier

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()

        pbar = tqdm(batchfier, total=len(batchfier.dataset))

        accuracy,_ = knn(model,"cuda",val_loader=pbar)


        return accuracy,0

    def ood_epoch(self, batchfier, train_features, train_y):
        model = self.model
        # batchfier = self.test_batchfier


        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.batch_size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)
        model.zero_grad()
        pbar = tqdm(batchfier, total=len(batchfier.dataset))
        pbar_cnt = 0
        tot_score = 0.0

        ood_for_save = {"m_distance": [],"fx": []}
        ood_features,_=get_features(model,batchfier)

        mahalanobis_score=get_scores_multi_cluster(train_features,ood_features,train_y)
        print(mahalanobis_score.shape)
        ood_for_save["m_distance"].extend(mahalanobis_score.tolist())
        ood_for_save["fx"].extend(ood_features.tolist())

        accuracy, _ = knn(model, "cuda", val_loader=pbar)


        for key,value in ood_for_save.items():

            print("{0} : {1}".format(key,len(value)))

        return pd.DataFrame(ood_for_save),accuracy
