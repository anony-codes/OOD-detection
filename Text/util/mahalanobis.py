import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader,Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import faiss
from tqdm import tqdm

# from util.batch_generator import BatchFierForOOD,BatchFierForMeasure


@torch.no_grad()
def sample_estimator(model, num_classes, feature_list, train_loader,device,args,class_wise=False):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    if isinstance(train_loader, IterableDataset):
        train_loader = DataLoader(dataset=train_loader,
                               batch_size=train_loader.batch_size,
                               shuffle=False,
                               collate_fn=train_loader.collate, pin_memory=True)

    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in tqdm(train_loader,total=len(train_loader.dataset)):
        data=data
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision,precision



def get_feature_list(model, flag_synthetic, device):
    if flag_synthetic:
        temp_x = torch.randint(low=0,high=10,size=[2,2]).to(device)
    else:
        temp_x = torch.rand(2, 3, 32, 32).to(device)
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    print([t.shape for t in temp_list])
    num_output = len(temp_list)
    feature_list = np.empty(num_output)

    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    return feature_list


@torch.no_grad()
def get_Mahalanobis_score(out_features, num_classes, sample_mean, precision, args):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    # model.eval()

    # out_features = model.intermediate_forward(input)
    # compute Mahalanobis score
    # out_features=fx

    n_batch = out_features[0].size(0)
    n_layer = len(sample_mean)

    gaussian_score = torch.zeros(n_batch, 2).to("cuda")

    for l in range(n_layer):
        out_feature=out_features[l]
        out_feature = out_feature.view(out_feature.size(0), -1)
        for i in range(num_classes):
            batch_sample_mean = sample_mean[l][i]
            zero_f = out_feature.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[l]), zero_f.t()).diag()
            gaussian_score[:,i]+=term_gau.view(-1)

    return gaussian_score


@torch.no_grad()
def get_Mahalanobis_score_penulti(out_features, num_classes, sample_mean, precision, args):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    # model.eval()

    # out_features = model.intermediate_forward(input)
    # compute Mahalanobis score
    # out_features=fx

    out_features = out_features[-1].view(out_features[-1].size(0), -1)
    gaussian_score = 0

    for i in range(num_classes):
        batch_sample_mean = sample_mean[-1][i]
        zero_f = out_features.data - batch_sample_mean

        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[-1]), zero_f.t()).diag()

        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

    return gaussian_score



def get_scores_multi_cluster(ftrain, target,ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    score = [
        np.sum(
            (target - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (target - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    score = np.min(score, axis=0)

    return score

def get_scores(ftrain, ftest, n_cluster):
    ypred = get_clusters(ftrain, n_cluster)
    return get_scores_multi_cluster(ftrain, ftest,ypred)

def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_features(model, dataloader):

    if isinstance( dataloader, IterableDataset):
        dataloader = DataLoader(dataset= dataloader,
                               batch_size= dataloader.batch_size,
                               shuffle=False,
                               collate_fn= dataloader.collate, pin_memory=True)

    features, labels = [], []
    model.eval()

    for inp, label in tqdm(dataloader, total=len(dataloader.dataset)):
        inp, label = inp.cuda(), inp.cuda()

        _,_,hidden = model(inp)

        features += list(hidden.data.cpu().numpy())
        labels += list(label.data.cpu().numpy())
        # total += len(img)

    features=np.array(features)
    features/=np.linalg.norm(features,axis=-1,keepdims=True)+1e-10

    return features, np.array(labels)