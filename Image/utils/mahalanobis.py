"""
Created on Sun Oct 21 2018
"""
from __future__ import print_function
import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


features = []

def get_features_hook(self, input, output):
    global features
    tmp_output = output.view(output.size(0), output.size(1), -1)
    tmp_output = torch.mean(tmp_output.data, 2)
    features.append(tmp_output)

def get_trg_layer(model, args):
    if (args.loss == 'ce') | (args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd')| (args.loss == 'ce_with_svd') |('uniform' in args.loss):
        trg_layer = model.fc[1]
    elif 'lb' in args.loss:
        trg_layer = model.fc[1]
    elif args.loss == 'final':
        trg_layer = model.fc[1]
    elif (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative') | (args.loss == 'ce_simclr_with_mu') | (
            args.loss == 'ce_simclr_negative_with_mu'):
        trg_layer = model.h[1]
    else:
        raise NotImplementedError
    return trg_layer

def get_multiple_trg_layer(model, args):
    if args.metric == 'mahalanobis_ensemble':
        if 'simclr' in args.loss:
            trg_lst = [model.encoder[3],
                       model.encoder[4],
                       model.encoder[5],
                       model.encoder[6],
                       model.encoder[7], ]
        else:
            trg_lst = [model.maxpool,
                       model.layer1,
                       model.layer2,
                       model.layer3,
                       model.layer4, ]
    else:
        trg_lst = []

    if (args.loss == 'ce') | (args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd')| (args.loss == 'ce_with_svd') :
        trg_layer = model.fc[1]
        trg_lst.append(trg_layer)
    elif 'lb' in args.loss:
        trg_layer = model.fc[1]
        trg_lst.append(trg_layer)
    elif 'uniform' in args.loss:
        trg_layer = model.fc[1]
        trg_lst.append(trg_layer)
    elif args.loss == 'final':
        trg_layer = model.fc[1]
        trg_lst.append(trg_layer)
    elif (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative') | (args.loss == 'ce_simclr_with_mu') | (
            args.loss == 'ce_simclr_negative_with_mu'):
        trg_layer = model.h[1]
        trg_lst.append(trg_layer)
    else:
        raise NotImplementedError
    return trg_lst

def get_multiple_feature_list(model, args, device):
    global features
    features = []
    temp_x = torch.rand(2,3,200,200).to(device)
    temp_x = Variable(temp_x)
    trgs = get_multiple_trg_layer(model, args)
    handles = []
    for trg_layer in trgs:
        handles.append(trg_layer.register_forward_hook(get_features_hook))
    model(temp_x)
    for handle in handles:
        handle.remove()
    temp_list = features

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    return feature_list

def sample_estimator(model, args, num_classes, feature_list, train_loader, device, pred_type, flag_sample_norm=False, flag_mean_norm=False, sphere_size = None, flag_simclr = None):
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
    num_sample_per_class = np.zeros(num_classes)
    list_features = np.zeros((num_output, num_classes)).tolist()

    for data, target in train_loader:
        total += len(target)
        data = data.to(device)
        data = Variable(data, requires_grad=False)

        trg_layers = get_multiple_trg_layer(model, args)
        handles = []
        global features
        features = []
        for trg_layer in trg_layers:
            handles.append(trg_layer.register_forward_hook(get_features_hook))

        if flag_simclr:
            _, _, output = model(data)
        else:
            output = model(data)
        for handle in handles:
            handle.remove()
        out_features = features
    
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            if flag_sample_norm:
                out_features[i] = F.normalize(out_features[i], p=2, dim=-1)*sphere_size

            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.to(device)).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(target.size(0)):
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
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
            if flag_mean_norm:
                temp_list[j] = F.normalize(temp_list[j], p=2, dim=-1)*sphere_size
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
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        if pred_type == 'diagonal':
            temp_inv_variance = torch.inverse(torch.diag(torch.var(X.cpu(), dim=0) + 1e-8)).numpy()
            temp_inv_variance = torch.from_numpy(temp_inv_variance).float().to(device)
            precision.append(temp_inv_variance)
        else:
            precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, args, data, num_classes, sample_mean, precision, layer_index=0, device=None, pred_type = 'org', normalized = False,
                          sphere_size = None):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
#     handle = model.layer4[1].bn2.register_forward_hook(get_features_hook)
    # handle = model.fc[0].register_forward_hook(get_features_hook)
    trg_layers = get_multiple_trg_layer(model, args)
    handles = []
    global features
    features = []
    for trg_layer in trg_layers:
        handles.append(trg_layer.register_forward_hook(get_features_hook))
    model(data)
    for handle in handles:
        handle.remove()
    # global features

    # compute Mahalanobis score
    tot_gaussian_score = None
    for layer_index in range(len(trg_layers)):
        out_features = features[layer_index]
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        if tot_gaussian_score is None:
            tot_gaussian_score = gaussian_score
        else:
            tot_gaussian_score += gaussian_score

    final_gaussian_score, pred_idx = torch.max(tot_gaussian_score, dim=1)

    return final_gaussian_score.data.cpu(), pred_idx

