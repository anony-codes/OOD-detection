import torch.nn as nn
import torchvision.models as models
from .simclr import SimCLR



def define_model(args, device='cuda'):
    if (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
        model = SimCLR(args, classes=args.num_classes, max_epoch=args.epochs)
    else:
        model = models.resnet18(pretrained=True)
        #     model = models.resnet18(pretrained = False)
        # model.fc =  nn.Linear(in_features=512, out_features=2, bias=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=128, bias=True),  # 0
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2, bias=True),  # 3
        )
        # model.fc = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=2, bias=True),  # 0
        # )
        if args.loss == 'ovadm':
            nn.init.xavier_uniform_(model.fc[2].weight)
        model = model.to(device)
    return model


