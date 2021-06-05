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

def get_multiple_trg_layer(model):
    trg_lst = [model.maxpool,
                       model.layer1,
                       model.layer2,
                       model.layer3,
                       model.layer4, ]
    trg_layer = model.fc[1]
    trg_lst.append(trg_layer)
    # print("TARGET LIST", trg_lst)
    return trg_lst

def get_multiple_feature_list(model, device):
    global features
    features = []
    temp_x = torch.rand(2,3,200,200).to(device)
    temp_x = Variable(temp_x)
    trgs = get_multiple_trg_layer(model)
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


def sample_estimator_ensemble(model, num_classes, feature_list, train_loader, device):
    """
    compute sample mean and precision (inverse of covariance) from Training dataset
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

        trg_layers = get_multiple_trg_layer(model)
        handles = []
        global features
        features = []
        for trg_layer in trg_layers:
            handles.append(trg_layer.register_forward_hook(get_features_hook))

        output = model(data)
        for handle in handles:
            handle.remove()
        out_features = features

        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
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
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


@torch.no_grad()
def get_Mahalanobis_score_ensemble(model, data, num_classes, sample_mean, precision, layer_index=0, device=None):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    trg_layers = get_multiple_trg_layer(model)
    handles = []
    global features
    features = []
    for trg_layer in trg_layers:
        handles.append(trg_layer.register_forward_hook(get_features_hook))
    model(data)
    for handle in handles:
        handle.remove()

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


