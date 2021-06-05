import torch.nn as nn
import torchvision.models as models


def define_model(args, device='cuda'):
    model = models.resnet18(pretrained=True)
    #     model = models.resnet18(pretrained = False)
    # model.fc =  nn.Linear(in_features=512, out_features=2, bias=True)
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=128, bias=True),  # 0
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10, bias=True),  # 3
    )
    # if args.loss == 'ovadm':
    #     nn.init.xavier_uniform_(model.fc[2].weight)
    model = model.to(device)
    return model



def calc_loss(model, criterion, inputs, labels, num_classes=10):
    loss, loss_ce, loss_reg, loss_pos, loss_neg = None, None, None, None, None
    # loss_mu, loss_var = None, None
    # penulti_ftrs, outputs = None, None

    outputs, penulti_ftrs = get_features(model, inputs, num_classes)
    loss = criterion(penulti_ftrs, labels)

    return loss


def get_features_hook(self, input, output):
    global features
    features = [output]

def get_features(model, data, num_classes):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    handle = model.fc[1].register_forward_hook(get_features_hook)
    out = model(data)
    handle.remove()
    global features
    out_features = features[0]

    return out, out_features



import sklearn.metrics as skm
import numpy as np
def get_roc_sklearn(xin, xood):
    xin=np.array(xin)
    xood=np.array(xood)
    # xin = xin[criteria]
    # xood = xood[criteria]

    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc