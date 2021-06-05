#[age]_[gender]_[race]_[date&time].jpg
import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def get_simclr_data_transforms(mode):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    if mode == "train":
        s=0.5
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=200),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * 32))], p=0.5),
                                              transforms.ToTensor()])
        return data_transforms
    else:
        data_transforms = transforms.Compose([transforms.Resize(200),
                                              transforms.ToTensor()])
        return data_transforms

class UTKDataset_Simclr(Dataset):
    def __init__(self, mode = 'train',  output_type = 'single', aug_type = 'basic'):
        X_train = np.load('./data/train_pths.npy')
        X_val = np.load('./data/val_pths.npy')
        X_test = np.load('./data/test_[26]_idx15.npy')

        transform_train = transforms.Compose(
            [
            transforms.Resize(200),
            transforms.RandomCrop(200, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
            )

        transform_test = transforms.Compose(
            [
            transforms.Resize(200),
            transforms.ToTensor(),
            ]
            )
        self.output_type = output_type
        self.aug_type = aug_type

        if mode == 'train':
            self.x_data = X_train
            self.y_data = np.array([int(pth.split('_')[2]) for pth in X_train])
            self.transform = transform_train
            if self.aug_type=='simclr':
                self.transform = get_simclr_data_transforms(mode='train')
            elif self.aug_type == 'basic':
                self.transform = transform_train
            elif self.aug_type == 'none':
                self.transform = transform_test
            else:
                raise NotImplementedError
        elif mode == 'val':
            self.x_data = X_val
            self.y_data = np.array([int(pth.split('_')[2]) for pth in X_val])
            self.transform = transform_test
        elif mode == 'test':
            self.x_data = X_test
            self.y_data = np.array([int(pth.split('_')[2]) for pth in X_test])
            self.transform = transform_test
        else:
            print("no matched version")


    def __getitem__(self, index):
        if self.output_type == 'tuple':
            x = (self.transform(Image.open(self.x_data[index]).convert('RGB')),
                 self.transform(Image.open(self.x_data[index]).convert('RGB')))
        elif self.output_type == 'single':
            x = self.transform(Image.open(self.x_data[index]).convert('RGB'))
        else:
            raise NotImplementedError
        y = self.y_data[index]
        return x,y

    def __len__(self):
        return self.x_data.shape[0]
    
    
class UTKDataset_test(Dataset):
    def __init__(self, age = '26'):
        if age == '26':
            X_test = np.load('./data/test_[26]_idx15.npy')
        else:
            X_test = np.array(list(filter((lambda name : name.split('/')[-1].split('_')[0] == age),
                                          sorted(glob.glob("./UTKFace/*")))))
        Y_test = np.array([int(pth.split('_')[2]) for pth in X_test])
              
        self.x_data = X_test
        self.y_data = Y_test
        self.transform = transforms.Compose(
            [
            transforms.Resize(200),
            transforms.ToTensor(),
            ]
            )
            
    def __getitem__(self, index):
        x = self.transform(Image.open(self.x_data[index]).convert('RGB'))
        y = self.y_data[index]
        return x,y

    def __len__(self):
        return self.x_data.shape[0]
    
class UTKDataset_200(Dataset):
    def __init__(self, age = '26'):
        if age == '26':
            X_test = np.load('./data/test_[26]_idx15.npy')

        else:
            pth = list(filter((lambda x: str(age) in x), glob.glob('./data/*')))[0]
            X_test = np.load(pth)
            
        Y_test = np.array([int(pth.split('_')[2]) for pth in X_test])
              
        self.x_data = X_test
        self.y_data = Y_test
        self.transform = transforms.Compose(
            [
            transforms.Resize(200),
            transforms.ToTensor(),
            ]
            )
            
    def __getitem__(self, index):
        x = self.transform(Image.open(self.x_data[index]).convert('RGB'))
        y = self.y_data[index]
        return x,y

    def __len__(self):
        return self.x_data.shape[0]


class UTKDataset_adjust(Dataset):
    def __init__(self, adjust_type='bright', adjust_scale=1):
        X_test = np.load('./data/test_[26]_idx15.npy')
        Y_test = np.array([int(pth.split('_')[2]) for pth in X_test])

        self.x_data = X_test
        self.y_data = Y_test

        self.transform = transforms.Compose(
            [
                transforms.Resize(200),
                transforms.ToTensor(),
            ]
        )

        self.adjust_type = adjust_type
        self.adjust_scale = adjust_scale

    def __getitem__(self, index):
        x = Image.open(self.x_data[index]).convert('RGB')
        if self.adjust_type == 'bright':
            x = transforms.functional.adjust_brightness(x, brightness_factor=self.adjust_scale)
            x = self.transform(x)
        elif self.adjust_type == 'contrast':
            x = transforms.functional.adjust_contrast(x, contrast_factor=self.adjust_scale)
            x = self.transform(x)
        else:
            print("unavailable adjust_type")
        y = self.y_data[index]
        return x, y

    def __len__(self):
        return self.x_data.shape[0]
    
def get_adjust_dataloaders(adjust = 'bright'):
    
    if adjust == 'bright':
#         adjust_scale = np.arange(1,5.5,0.25)
        adjust_scale = [1]
        adjust_scale += list(np.arange(0,10,1)/10)
        adjust_scale += list(np.arange(10,50,5)/10)
        # adjust_scale += list(np.arange(50,350,50)/10)
        adjust_scale += [5.0, 10.0]
        
    elif adjust == 'contrast':
#         adjust_scale = np.arange(1,11.5,0.5)
        adjust_scale = [1]
        adjust_scale += list(np.arange(0,10,1)/10)
        adjust_scale += list(np.arange(10,50,5)/10)
        # adjust_scale += list(np.arange(50,350,50)/10)
        adjust_scale += [5.0, 10.0]
           
    loaders = []
    data_len = []
    ind = UTKDataset_adjust(adjust_type = adjust, adjust_scale = 1)
    ind_len = len(ind)
    data_len.append(ind_len)
    ind_loader = DataLoader(ind, batch_size=64, shuffle=False, num_workers=8)
    loaders.append(ind_loader)
    
    for adjustness in adjust_scale[1:]:
        ood = UTKDataset_adjust( adjust_type = adjust, adjust_scale = adjustness)
        ood_len = len(ood)
        data_len.append(ood_len)
        ood_loader = DataLoader(ood, batch_size=64, shuffle=False, num_workers=8)
        loaders.append(ood_loader)
    return loaders, data_len, adjust_scale

def get_eval_200_dataloaders():
    ages = [[1],
             [2],
             [3, 4],
             [5, 6],
             [7, 8],
             [9, 10, 11],
             [12, 13, 14],
             [15, 16],
             [17, 18],
             [19, 20],
             [21],
             [22],
             [23],
             [24],
             [25],
#              [26],
           ]
    ages.reverse()
    
    loaders = []
    data_len = []
    ind = UTKDataset_test(age = '26')
    ind_len = len(ind)
    data_len.append(ind_len)
    ind_loader = DataLoader(ind, batch_size=64, shuffle=False, num_workers=8)
    loaders.append(ind_loader)
    
    for age in ages:
        ood = UTKDataset_200(age = age)
        ood_len = len(ood)
        data_len.append(ood_len)
        ood_loader = DataLoader(ood, batch_size=64, shuffle=False, num_workers=8)
        loaders.append(ood_loader)
    return loaders, data_len

def get_loaders(output_type = None, aug_type = None, args=None):
    train_dataset = UTKDataset_Simclr(mode='train', output_type=output_type, aug_type=aug_type)
    val_dataset = UTKDataset_Simclr(mode='val', output_type=output_type)
    test_dataset = UTKDataset_Simclr(mode='test', output_type=output_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader_mu = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)  # org
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True) # org
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    return train_loader, train_loader_mu, val_loader, test_loader
