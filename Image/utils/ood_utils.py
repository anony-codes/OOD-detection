import random
import os
from .mahalanobis import *


def get_outputs(model, images, args):
    outputs = model(images)
    logit = outputs
    outputs = torch.nn.Softmax(dim=1)(outputs)
    return logit, outputs


def get_ood_outputs(model, images, args, stat_mahala=None, device=None, pred_type = 'org'):
    if (args.metric == 'mahalanobis') | (args.metric == 'mahalanobis_ensemble'):
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
    if args.flag_v3:
        args.result_path = os.path.join(args.result_path, "v3")
    if args.flag_v2:
        args.result_path = os.path.join(args.result_path, "v2")
    if args.flag_entropy:
        args.result_path = os.path.join(args.result_path, "entropy")

    if args.flag_entropy_clswise:
        args.result_path = os.path.join(args.result_path, "entropy_cls")

    if 'lb' in args.loss:
        args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "smoothing_{}".format(args.smooth_factor))
    elif 'simclr' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_{}".format(args.M), "lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))
            else:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu),"lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))
        else:
            args.result_path = os.path.join(args.result_path, args.loss, args.aug_type,"lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))
    elif 'svd' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_{}".format(args.M),"lambda_var_{}".format(args.lambda_loss_var))
            else:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu),"lambda_var_{}".format(args.lambda_loss_var))
        else:
            args.result_path = os.path.join(args.result_path, args.loss, args.aug_type,"lambda_var_{}".format(args.lambda_loss_var))

    elif 'uniform' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_{}".format(args.M),"lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))
            else:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu),"lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))
        else:
            args.result_path = os.path.join(args.result_path, args.loss, args.aug_type,"lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature))

    elif args.loss == 'final':
        args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "lambda_mu_{}".format(args.lambda_loss_mu), "lambda_var_{}".format(args.lambda_loss_var), "lambda_corr_{}".format(args.lambda_loss_corr))
    else:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_{}".format(args.M))
            elif args.flag_entropy:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None",
                                                "lambda_{}".format(args.lambda_loss_mu), "lambda_var_{}".format(args.lambda_loss_var))
            elif args.flag_entropy_clswise:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None",
                                                "lambda_{}".format(args.lambda_loss_mu),
                                                "lambda_var_{}".format(args.lambda_loss_var))
            else:
                args.result_path = os.path.join(args.result_path, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu))
        else:
            args.result_path = os.path.join(args.result_path, args.loss, args.aug_type)
    return args.result_path


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)