import cv2
import glob
import torch
import random
import numpy as np 
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from mahalanobis import *
from mahalanobis_ensemble import *
from odin import *
from dataset import *
from skimage import io, transform
from torchvision import datasets, models, transforms

def define_model(device = 'cuda'):
    model = models.resnet18(pretrained = True)
    model.fc = nn.Sequential(
              nn.Linear(in_features = 512, out_features = 128, bias = True),
              nn.ReLU(),
              nn.Linear(in_features = 128, out_features = 2, bias = True),
            )
    model = model.to(device)
    return model

def get_outputs(model, images, args):
    outputs = model(images)
    logit = outputs
    outputs = nn.Softmax()(outputs)
    return logit, outputs

def get_ood_outputs(model, images, args, outputs = None, stat_mahala = None, criterion = None, epsilon = None,  device = None):    
    if args.metric == 'mahalanobis': 
        mean, var = stat_mahala
        predicted_ood, preds = get_Mahalanobis_score(model, images, num_classes = args.num_classes, sample_mean = mean, precision = var, layer_index = 0, device = device)
        
        # return predicted_ood, preds
    elif args.metric == 'odin':
        predicted_ood = get_odin_score(args, model, images, 1000, epsilon, criterion = None, device = device)
    if args.metric == 'mahalanobis_ensemble': 
        mean, var = stat_mahala
        predicted_ood, preds = get_Mahalanobis_score_ensemble(model, images, num_classes = args.num_classes, sample_mean = mean, precision = var, layer_index = 0, device = device)
        
    return predicted_ood

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Data_Transform(path):
    k = 100
    size = 500
    train_dataset_path =  os.path.join(path, 'boneage-training-dataset/boneage-training-dataset/')
    csv_path =  os.path.join(path, 'boneage-training-dataset.csv')
    ############## Find mean and std of dataset###################
    image_filenames = glob.glob(train_dataset_path+'*.png')
    random_images = random.sample(population = image_filenames,k = len(image_filenames))

#     means = []
#     stds = []

#     for filename in random_images:
#         image = cv2.imread(filename,0)
#         image = cv2.resize(image,(size,size))
#         mean,std = cv2.meanStdDev(image)
#     #    mean /= 255
#     #    std /= 255
        
#         means.append(mean[0][0])
#         stds.append(std[0][0])

#     avg_mean = np.mean(means) 
#     avg_std = np.mean(stds)

#     print('Approx. Mean of Images in Dataset: ',avg_mean)
#     print('Approx. Standard Deviation of Images in Dataset: ',avg_std)

    ############### After calculating mean and std of entire dataset from above, we get the below values ############
    avg_mean = 46.49
    avg_std = 42.56
    
    dataset_size = len(image_filenames)-2800
    val_size = dataset_size + 1400

    bones_df = pd.read_csv(csv_path)
    bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)
    bones_df['AgeM']=bones_df['boneage'].apply(lambda x: round(x/12.)).astype(int) #converting age from months to years

    train_df = bones_df.iloc[:dataset_size,:]
    val_df = bones_df.iloc[dataset_size:val_size,:]
    test_df = bones_df.iloc[val_size:,:]

    age_max = np.max(bones_df['boneage'])
    age_min = np.min(bones_df['boneage'])
    print("avg_mean,avg_std,age_min,age_max", avg_mean,avg_std)
    data_transform = transforms.Compose([
    Normalize(avg_mean,avg_std,age_min,age_max),
    ToTensor()   
    ]) 
    return bones_df, train_df, val_df, test_df, data_transform