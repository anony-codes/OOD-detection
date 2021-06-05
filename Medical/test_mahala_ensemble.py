import numpy as np
import os
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

from utils import *
from dataset import *
from mahalanobis_ensemble import *
from odin import *
from main import *

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = None

def test(model, args, train_loader = None, val_loader = None, loaders = None, device = None):
    model = model.to(device)
    model.eval()

    criterion = define_criterion(args, model)

    total = 0
    correct = 0
    dict_results = dict()
    stat_results = dict()
    dict_results['preds'] = []
    dict_results['trues'] = []
    dict_results['correct'] = []
    dict_results['dataset_idx'] = []
    dict_results['org_labels'] = []
    dict_results['pred_labels'] = []

    if (args.loss == 'ce') & (args.metric == 'baseline'):
        dict_results['checksum'] = []
        dict_results['full_preds'] = []
        
    test_loaders = loaders
    print(len(test_loaders))
    
    if args.metric == 'mahalanobis_ensemble':
        feature_list = get_multiple_feature_list(model, device=device)
        print(feature_list)

        mean, var = sample_estimator_ensemble(model, num_classes=args.num_classes, feature_list= feature_list, train_loader = train_loader, device = device)

        stat_results['mean'] = mean
        stat_results['var'] = var
        print("success")


    if args.flag_adjust:
        n = 0
    else:
        n = 5

    if(args.metric == 'baseline' or args.metric == 'mahalanobis_ensemble'):
        with torch.no_grad():
            for idx, loader in enumerate(test_loaders):
                value_ind = None
                value_ood = None
                for data in loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    labels = labels.squeeze()
                    if idx == n:
                        labels_ood = torch.ones(labels.size(0))
                    else:
                        labels_ood = torch.zeros(labels.size(0))
                    labels_ood = labels_ood.to(device)
                    
                    outputs, penulti_ftrs = get_features(model, images, args.num_classes)
                    predicted, predicted_value = criterion.predict(penulti_ftrs)

                    #get ood statistics
                    if args.metric == 'baseline':
                        if args.loss =='ce':
                            dict_results['full_preds'] += (predicted_value).tolist()
                            predicted_ood, _ = torch.max(predicted_value, dim=-1)
                        else:
                            predicted_ood = predicted_value
                    elif args.metric == 'mahalanobis_ensemble':
                        predicted_ood = get_ood_outputs(model, images, args, stat_mahala = (mean, var), device = device)

                    else:
                        print('metric is not available')

                    if idx == n:
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                    dict_results['preds']+=(predicted_ood).tolist()
                    dict_results['trues']+=(labels_ood).tolist()
                    dict_results['org_labels']+=(labels).tolist()
                    dict_results['pred_labels']+=(predicted).tolist()
                    if (args.loss == 'ce') & (args.metric == 'baseline'):
                        dict_results['checksum'] += (torch.sum(images.flatten(1), dim = 1)).tolist()
                    if idx ==n:
                        dict_results['correct']+=(predicted == labels).tolist()
                    else:
                        dict_results['correct']+=(torch.zeros_like(labels)-1).tolist()
                    dict_results['dataset_idx']+=(torch.zeros_like(labels)+idx).tolist()

                    if idx == n:
                        if value_ind is None:
                            value_ind = predicted_ood
                        else:
                            value_ind = torch.cat([value_ind, predicted_ood])
                    else:
                        if value_ood is None:
                            value_ood = predicted_ood
                        else:
                            value_ood = torch.cat([value_ood,predicted_ood])
    else:
        for idx, loader in enumerate(test_loaders):
            value_ind = None
            value_ood = None
            for data in loader:
                images1, labels = data
                images = images1.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                if idx == n:
                    labels_ood = torch.ones(labels.size(0))
                else:
                    labels_ood = torch.zeros(labels.size(0))
                labels_ood = labels_ood.to(device)

                outputs, penulti_ftrs = get_features(model, images, args.num_classes)
                predicted, predicted_value = criterion.predict(penulti_ftrs)

                #get ood statistics
                if args.metric == 'baseline':
                    if args.loss =='ce':
                        dict_results['full_preds'] += (predicted_value).tolist()
                        predicted_ood, _ = torch.max(predicted_value, dim=-1)
                    else:
                        predicted_ood = predicted_value
                elif args.metric == 'mahalanobis_ensemble':
                    predicted_ood = get_ood_outputs(model, images, args, stat_mahala = (mean, var), device = device)
                elif args.metric == 'odin':
                    predicted_ood = get_ood_outputs(model, images, args, outputs = None, stat_mahala = None, criterion = None, epsilon = 0.005, device = device)

                else:
                    print('metric is not available')

                if idx == n:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                dict_results['preds']+=(predicted_ood).tolist()
                dict_results['trues']+=(labels_ood).tolist()
                dict_results['org_labels']+=(labels).tolist()
                dict_results['pred_labels']+=(predicted).tolist()
                if (args.loss == 'ce') & (args.metric == 'baseline'):
                    dict_results['checksum'] += (torch.sum(images.flatten(1), dim = 1)).tolist()
                if idx ==n:
                    dict_results['correct']+=(predicted == labels).tolist()
                else:
                    dict_results['correct']+=(torch.zeros_like(labels)-1).tolist()
                dict_results['dataset_idx']+=(torch.zeros_like(labels)+idx).tolist()

                if idx == n:
                    if value_ind is None:
                        value_ind = predicted_ood
                    else:
                        value_ind = torch.cat([value_ind, predicted_ood])
                else:
                    if value_ood is None:
                        value_ood = predicted_ood
                    else:
                        value_ood = torch.cat([value_ood,predicted_ood])



    print(correct, total)
    acc = (100 * correct / total)
    dict_results['acc'] = acc
    print('Accuracy of the network on the test images: {} %%'.format(acc))
    
    if args.flag_adjust:
        dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed))
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "scores_adjust_{}.pkl".format(args.type_adjust))
    else:
        dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed))
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "scores_age.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump(dict_results, f)
  
    if 'mahalanobis_ensemble' in args.metric:
        stat_path = os.path.join(dir_path, "mean_var.pkl")
        with open(stat_path, "wb") as f:
            pickle.dump(stat_results, f)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, help='mahalanobs or baseline')
    parser.add_argument('--flag_adjust', action='store_true', help='adjust test or not')
    parser.add_argument('--type_adjust', type=str, help='path of the model')
    parser.add_argument('--num_classes', default = 2, type=int, help='path of the model')
    parser.add_argument('--result_path', default="./results", type=str, help='train or test')
    parser.add_argument('--seed', default=0, type=int, help='train or test')
    parser.add_argument('--loss', default="CEwithMuVarCorr", type=str, help='train or test')
    parser.add_argument('--w1', default=1.0, type=float, help='weightage for CE loss')
    parser.add_argument('--w2', default=1.0, type=float, help='weightage for MU loss')
    parser.add_argument('--w3', default=1.0, type=float, help='weightage for variance loss')
    parser.add_argument('--w4', default=1.0, type=float, help='weightage for entropy loss')
    parser.add_argument('--data_path', default="./data", type=str, help='path of the dataset')

    args = parser.parse_args()
    set_seed(args.seed)

    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    bones_df, train_df, val_df, test_df, data_transform = Data_Transform()
    age_groups = [[1,2,3,4,5],[6],[7],[8],[9],[10,11,12],[13],[14],[15,16,17,18,19]]
    
    ##############################
    # Data
    ##############################
    images_dir = os.path.join(args.data_path, 'boneage-training-dataset/boneage-training-dataset/') 
    if args.flag_adjust:
        print("true")
        loaders, data_len, _   = get_adjust_dataloaders(bones_df, train_df, val_df, test_df, data_transform, images_dir, args.type_adjust)
    else:
        print("false")
        loaders, data_len = get_eval_dataloaders(bones_df, train_df, val_df, test_df, data_transform,images_dir, age_groups)
    train_dataset = BoneDataset(dataframe = train_df,img_dir=images_dir, mode = 'train', transform = data_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = BoneDataset(dataframe = val_df, img_dir=images_dir, mode = 'val', transform = data_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    ##############################
    # Model
    ##############################
    print(args.seed)
    model = define_model(device)
    model.to(device)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    
    
    ###############################
    # Test
    ###############################
    test(model, args, train_loader = train_loader, val_loader = val_loader, loaders = loaders, device = device)
            
            
if __name__ == '__main__':
    main()