import random
import os
from .mahalanobis import *

def get_trg_layer(model, args):
    if (args.loss == 'ce')| (args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd') | (args.loss == 'ce_with_svd') :
        trg_layer = model.fc[1]
    elif 'uniform' in args.loss:
        trg_layer = model.fc[1]
    elif 'final' in args.loss:
        trg_layer = model.fc[1]
    elif args.loss == 'ce_simclr':
        trg_layer = model.h[1]
    elif args.loss == 'ce_simclr_negative':
        trg_layer = model.h[1]
    else:
        raise NotImplementedError
    return trg_layer

def get_outputs(model, images, args):
    outputs = model(images)
    logit = outputs
    outputs = torch.nn.Softmax(dim=1)(outputs)
    return logit, outputs


def get_ood_outputs(model, images, args, stat_mahala=None, device=None, pred_type = 'org'):
    if args.metric == 'mahalanobis':
        mean, var = stat_mahala
        predicted_ood, _ = get_Mahalanobis_score(model, args, images, num_classes=args.num_classes, sample_mean=mean, precision=var, layer_index=0,
                                                 device=device, pred_type = pred_type)
    #         predicted_ood, _ = torch.max(ood_outputs.data, 1)
    elif (args.metric == 'mahalanobis_sample_norm') | (args.metric == 'mahalanobis_sample_mean_norm'):
        mean, var = stat_mahala
        predicted_ood, _ = get_Mahalanobis_score(model, args, images, num_classes=args.num_classes, sample_mean=mean, precision=var, layer_index=0,
                                                 device=device, pred_type=pred_type, normalized = True, sphere_size = args.S)
    else:
        raise NotImplementedError
    return predicted_ood


def get_result_path(args):
    args.result_path = os.path.join(args.result_path, args.loss, args.aug_type)
    return args.result_path


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)