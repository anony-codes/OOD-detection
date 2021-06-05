from dataloader import get_train_valid_loader, get_test_loader
import torch
from tqdm import tqdm
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.optim as optim
from model.utils import get_features, calc_loss

import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import argparse

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         penulti = F.relu(self.fc1(x))
#         out = self.fc2(penulti)
#         return out, penulti

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.fc3(x)

        return logits, x

def accuracy(model,loader, device):
    total = 0
    correct = 0
    model.eval()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = (100 * correct / total)

    return acc
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./results', help='path of the model')
    parser.add_argument('--loss', type=str, default='ce', help='path of the model')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--n_epoch', default=50, type=int, help='seed')
    parser.add_argument('--w1', default=0.0, type=float, help='seed')
    parser.add_argument('--w2', default=0.0, type=float, help='seed')
    parser.add_argument('--w3', default=0.0, type=float, help='seed')

    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    loss_type = args.loss
    w1, w2, w3 = args.w1, args.w2, args.w3
    seed = args.seed
    result_pth = args.root_path

    
    model_pth = os.path.join(result_pth, loss_type, "models")
    log_pth = os.path.join(result_pth, loss_type, "logs")
    if loss_type == 'ours':
        model_pth = os.path.join(model_pth, "{}_{}_{}".format(w1, w2, w3))
        log_pth = os.path.join(log_pth, "{}_{}_{}".format(w1, w2, w3))
    os.makedirs(model_pth, exist_ok = True)
    os.makedirs(log_pth, exist_ok = True)
    best_file = os.path.join(model_pth, "best_{}.pt".format(seed))
    log_file = os.path.join(log_pth, "log_{}.txt".format(seed))
    acc_file = os.path.join(log_pth, "test_acc_{}.txt".format(seed))

        
    ############################
    # Data
    ############################
    train_loader, val_loader = get_train_valid_loader(data_dir = './data', batch_size = 256, random_seed = 0)
    test_loader = get_test_loader(data_dir = './data', batch_size = 256)

    ##############################
    # Model
    ##############################
    # from model.resnet import WideResNet
    # define_model("cuda")
    import apex
    # model = Net().to("cuda")
    # model = WideResNet(40, 10, 2, dropRate=0.3)
    # model=WideResNet().to("cuda")
    from model.utils import define_model
    model=define_model(args)
    
    ###############################
    # Training
    ###############################
    if loss_type == 'ours':
        from model.losses import Ours
        criterion_ours = Ours(w1, w2, w3, device).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


    # model, optimizer = apex.amp.initialize(model,optimizer,opt_level="O2")

    best_acc = -1

    early_stop=0

    for epoch in tqdm(range(args.n_epoch)):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            # get_features(model,)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            if loss_type == 'ours':
                reg=calc_loss(model,criterion_ours,inputs,labels)
#                 print(reg)
                loss += reg

            loss.backward()
            optimizer.step()

        val_acc = accuracy(model, val_loader, device)

        table = 'Epoch: {}, Validation acc: {}'.format(epoch + 1, val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_file)
            table += "   <<< best acc"
            early_stop=0
        else:
            early_stop+=1

        if early_stop>=10:
            break

        print(table)
        with open(log_file, "a") as file:
            file.write(table)   

    print('Finished Training')

    test_acc = accuracy(model, test_loader, device)
    table = 'Test acc: {}'.format(test_acc)
    with open(acc_file, "a") as file:
        file.write(table)    
           
if __name__ == '__main__':
    main()
