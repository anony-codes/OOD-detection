import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import os
import numpy as np
import random
from model.utils import get_roc_sklearn

# from utils import define_model, get_model_pth, get_result_pth
from dataloader import get_train_valid_loader, get_test_loader, get_ood_loader_CIFAR100

# from odin import sample_odin_estimator, get_odin_score
from mahalanobis import get_feature_list, sample_estimator, get_Mahalanobis_score

# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
import torch.nn.init
# import torch.optim as optim
from model.losses import Ours
import torch.nn as nn
import torch.nn.functional as F


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        penulti = F.relu(self.fc1(x))
        out = self.fc2(penulti)
        return out, penulti


def test(model, args, ood_loader=None, odin_loader=None, mahala_train_loader=None, flag_ood=False):
    dict_results = dict()
    dict_results['data'] = []
    dict_results['score'] = []

        # if 'mahalanobis' in args.metric:
        #     feature_list = get_feature_list(model, args, device=args.device)
        #     mean, var = sample_estimator(model, args, num_classes=args.num_classes, feature_list=feature_list,
        #                                  train_loader=mahala_train_loader,
        #                                  device=args.device)

    feature_list = get_feature_list(model, device=args.device)
    mean, var = sample_estimator(model, num_classes=10, feature_list=feature_list,
                                 train_loader=mahala_train_loader,
                                 device=args.device)

    for data in tqdm(ood_loader):
        inputs, labels = data
        inputs = inputs.to(args.device)

        outputs = model(inputs)
        outputs = torch.nn.Softmax(dim=1)(outputs)
        predicted_value, predicted = torch.max(outputs.data, 1)

        # get ood statistics
        #         if args.metric == 'baseline':
        #             score = predicted_value

        #         elif 'mahalanobis' in args.metric:
        #             score, _ = get_Mahalanobis_score(model, args, inputs, num_classes=args.num_classes,
        #                                                      sample_mean=mean, precision=var)

        #         elif 'odin' in args.metric:
        #             score = get_odin_score(args, model, inputs, temperature=1000, epsilon=epsilon,
        #                                            criterion=odin_criterion)
        #         else:
        #             print('metric is not available')

        score, _ = get_Mahalanobis_score(model, inputs, num_classes=10,
                                         sample_mean=mean, precision=var)

        dict_results['data'] += inputs.tolist()
        dict_results['score'] += score.tolist()

    if flag_ood:
        #         file_path = os.path.join(args.result_path, 'scores(ood).pkl')
        file_path = os.path.join(args.result_path, 'scores(ood_mix).pkl')
    else:
        file_path = os.path.join(args.result_path, 'scores(test).pkl')

    with open(file_path, "wb") as f:
        pickle.dump(dict_results, f)

    print(min(dict_results["score"]))
    print(max(dict_results["score"]))

    return dict_results['score']


def test_mixup(model, args, ood_loader=None, test_loader=None, mahala_train_loader=None, flag_ood=False,
               mixup_ratio=0.1):
    print(mixup_ratio)
    dict_results = dict()
    dict_results['data'] = []
    dict_results['score'] = []
    #
    feature_list = get_feature_list(model, device=args.device)
    mean, var = sample_estimator(model, num_classes=10, feature_list=feature_list,
                                 train_loader=mahala_train_loader,
                                 device=args.device)

    for ood_data, test_data in tqdm(zip(ood_loader, test_loader), total=len(ood_loader)):
        ood_inputs, ood_labels = ood_data
        test_inputs, test_labels = test_data


        inputs = ood_inputs * mixup_ratio + test_inputs * (1 - mixup_ratio)
        inputs = inputs.to(args.device)

        outputs = model(inputs)
        outputs = torch.nn.Softmax(dim=1)(outputs)
        predicted_value, predicted = torch.max(outputs.data, 1)
        score, _ = get_Mahalanobis_score(model, inputs, num_classes=10,
                                         sample_mean=mean, precision=var)

        dict_results['data'] += inputs.tolist()
        dict_results['score'] += score.tolist()

    if flag_ood:
        #         file_path = os.path.join(args.result_path, 'scores(ood).pkl')
        file_path = os.path.join(args.result_path, 'scores(ood_mix).pkl')
    else:
        file_path = os.path.join(args.result_path, 'scores(test).pkl')

    with open(file_path, "wb") as f:
        pickle.dump(dict_results, f)

    # print(dict_results["score"][:100])

    print(min(dict_results["score"]))
    print(max(dict_results["score"]))

    return dict_results['score']


def main():
    parser = argparse.ArgumentParser()
    #     parser.add_argument('--data', type=str, help='type of data')
    #     parser.add_argument('--root_path', type=str, default='./results', help='root path')
    #     parser.add_argument('--loss', type=str, default='ce', help='loss')
    #     parser.add_argument('--metric', type=str, default='baseline', help='score metric')
    #     parser.add_argument('--seed', default=0, type=int, help='seed')
    #     parser.add_argument('--flag_indist', action='store_true', help='indistribution dataset')
    #     parser.add_argument('--flag_auroc', action='store_true', help='indistribution dataset')
    #     args = parser.parse_args()

    #     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     args.result_path = get_result_pth(args)
    #     args.num_classes = 10
    #     print(args)
    parser.add_argument('--root_path', type=str, default='./results', help='path of the model')
    parser.add_argument('--loss', type=str, default='ce', help='path of the model')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--w1', default=0.0, type=float, help='seed')
    parser.add_argument('--w2', default=0.0, type=float, help='seed')
    parser.add_argument('--w3', default=0.0, type=float, help='seed')

    args = parser.parse_args()
    args.num_classes = 10

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    loss_type = args.loss
    w1, w2, w3 = args.w1, args.w2, args.w3
    seed = args.seed
    result_pth = args.root_path

    model_pth = os.path.join(result_pth, loss_type, "models")
    result_pth = os.path.join(result_pth, loss_type, "aurocs")
    if loss_type == 'ours':
        model_pth = os.path.join(model_pth, "{}_{}_{}".format(w1, w2, w3))
        result_pth = os.path.join(result_pth, "{}_{}_{}".format(w1, w2, w3))
    os.makedirs(model_pth, exist_ok=True)
    os.makedirs(result_pth, exist_ok=True)
    best_file = os.path.join(model_pth, "best_{}.pt".format(seed))
    print(best_file)
    args.result_path = result_pth

    ##############################
    # Data
    ##############################
    train_loader, val_loader = get_train_valid_loader(data_dir='./data', batch_size=256, random_seed=0)
    test_loader = get_test_loader(data_dir='./data', batch_size=256)

    ood_loader = get_ood_loader_CIFAR100(data_dir='./data', batch_size=256)

    ##############################
    # Model
    ##############################
    from model.utils import define_model
    model = define_model(args)
    model.load_state_dict(torch.load(best_file))

    model.eval()

    # feature_list = get_feature_list(model, device=args.device)
    # mean, var = sample_estimator(model, num_classes=10, feature_list=feature_list,
    #                              train_loader=train_loader,
    #                              device=args.device)

    test_score = test(model, args, ood_loader=test_loader, odin_loader=None, mahala_train_loader=train_loader,
                      flag_ood=False)

    # ood_score = test(model, args, ood_loader=ood_loader, odin_loader=None, mahala_train_loader=train_loader,
    #                   flag_ood=False)

    auroc_list = []

    for i in range(0, 11):
        ood_score = test_mixup(model, args, ood_loader=ood_loader, test_loader=test_loader, mahala_train_loader=train_loader,
                               flag_ood=True, mixup_ratio=i / 10)
        auroc=get_roc_sklearn(test_score, ood_score)
        auroc_list.append(auroc)
        print(auroc)
    # auroc_list = [get_roc_sklearn(test_score, ood_score)]
    #

    #     auroc_list.append(auroc)

    # print(test_score)
    # print(ood_score)

    if loss_type == "ce":
        f = open(f"result_{loss_type}.txt", "w")
    else:
        f = open(f"result_{loss_type}_{w1}_{w2}_{w3}.txt", "w")

    for auroc in auroc_list:
        f.write(str(auroc) + "\n")

    f.close()


#     if args.loss == 'ce':
#         test(model, args, ood_loader=test_loader, odin_loader=None, mahala_train_loader=train_loader)
#     else:
#         test(model, args, ood_loader=ood_loader, odin_loader=None, mahala_train_loader=train_loader)

if __name__ == '__main__':
    main()
