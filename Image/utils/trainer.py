
import os

import torch.optim as optim
from tqdm import tqdm

from model.model_utils import *
from .ood_utils import *
from .dataset import *
from model.simclr import *
from model.losses import *
from model.loss_rescale import *

from tensorboardX import SummaryWriter
# import apex


features = None

def get_features_hook(self, input, output):
    global features
    features = [output]

def get_features(model, args, data):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    trg_layer = get_trg_layer(model, args)

    handle = trg_layer.register_forward_hook(get_features_hook)
    out = model(data)
    handle.remove()
    global features
    out_features = features[0]

    return out, out_features


def get_run_dir(run_dir, args):
    if args.flag_v3:
        run_dir = os.path.join(run_dir, "v3")
    if args.flag_v2:
        run_dir = os.path.join(run_dir, "v2")
    if args.flag_entropy:
        run_dir = os.path.join(run_dir, 'entropy')
    if args.flag_entropy_clswise:
        run_dir = os.path.join(run_dir, 'entropy_cls')
    
    if 'lb' in args.loss:
        run_dir = os.path.join(run_dir, args.loss, args.aug_type, "smoothing_{}".format(args.smooth_factor), "seed_{}".format(args.seed))
    elif 'simclr' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_{}".format(args.M), "lambda_var_{}".format(args.lambda_loss_var),"temp_{}".format(args.temperature), "seed_{}".format(args.seed))
            else:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu), "lambda_var_{}".format(args.lambda_loss_var),"temp_{}".format(args.temperature), "seed_{}".format(args.seed))
        else:
            run_dir = os.path.join(run_dir, args.loss, args.aug_type, "lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature), "seed_{}".format(args.seed))
    elif 'svd' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_{}".format(args.M), "lambda_var_{}".format(args.lambda_loss_var),"seed_{}".format(args.seed))
            else:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None", "lambda_mu_{}".format(args.lambda_loss_mu), "lambda_var_{}".format(args.lambda_loss_var), "seed_{}".format(args.seed))
        else:
            run_dir = os.path.join(run_dir, args.loss, args.aug_type, "lambda_var_{}".format(args.lambda_loss_var),"seed_{}".format(args.seed))
    elif 'uniform' in args.loss:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_{}".format(args.M), "lambda_var_{}".format(args.lambda_loss_var),"temp_{}".format(args.temperature),"seed_{}".format(args.seed))
            else:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None", "lambda_mu_{}".format(args.lambda_loss_mu), "lambda_var_{}".format(args.lambda_loss_var), "temp_{}".format(args.temperature),"seed_{}".format(args.seed))
        else:
            run_dir = os.path.join(run_dir, args.loss, args.aug_type, "lambda_var_{}".format(args.lambda_loss_var),"temp_{}".format(args.temperature),"seed_{}".format(args.seed))
    elif args.loss == 'final':
        run_dir = os.path.join(run_dir, args.loss, args.aug_type,
                                        "lambda_mu_{}".format(args.lambda_loss_mu),
                                        "lambda_var_{}".format(args.lambda_loss_var),
                                        "lambda_corr_{}".format(args.lambda_loss_corr), "seed_{}".format(args.seed))
    else:
        if 'with_mu' in args.loss:
            if args.flag_margin:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_{}".format(args.M), "seed_{}".format(args.seed))
            elif args.flag_entropy:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None",
                                       "lambda_{}".format(args.lambda_loss_mu),"lambda_var_{}".format(args.lambda_loss_var), "seed_{}".format(args.seed))
            elif args.flag_entropy_clswise:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None",
                                       "lambda_{}".format(args.lambda_loss_mu),
                                       "lambda_var_{}".format(args.lambda_loss_var), "seed_{}".format(args.seed))
            else:
                run_dir = os.path.join(run_dir, args.loss, args.aug_type, "margin_None", "lambda_{}".format(args.lambda_loss_mu), "seed_{}".format(args.seed))
        else:
            run_dir = os.path.join(run_dir, args.loss, args.aug_type, "seed_{}".format(args.seed))
    return run_dir

def simclr_train(args):
    #############################
    # define summary writer
    #############################
    run_dir = "./runs/"
    run_dir = get_run_dir(run_dir, args)
    os.makedirs(run_dir, exist_ok=True)
    summary = SummaryWriter(run_dir)

    #######################################################
    # define dataset , model, criterion & optimizer
    #######################################################
    train_loader, _, val_loader, test_loader = get_loaders(output_type = 'tuple', aug_type = args.aug_type, args=args)

    ##############################
    # training
    ##############################]
    simclr = define_model(args)
    simclr.train([train_loader, val_loader, test_loader], summary)


def calc_loss(model, criterion, inputs, labels, args, criterion_svd=None, criterion_entropy= None):
    loss=None
    loss_ce, loss_mu = None, None
    loss_svd, loss_entropy = None, None
    penulti_ftrs, outputs = None, None

    if args.loss == 'ce':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss = criterion(penulti_ftrs, labels)
    if args.loss == 'ce_with_mu':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss, loss_ce, loss_mu = criterion(penulti_ftrs, labels)
    if args.loss == 'ce_with_mu_svd':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss, loss_ce, loss_mu = criterion(penulti_ftrs, labels)
        loss_svd = criterion_svd(penulti_ftrs, labels)
        loss += loss_svd
    if args.loss == 'ce_with_svd':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss = criterion(penulti_ftrs, labels)
        loss_svd = criterion_svd(penulti_ftrs, labels)
        loss += loss_svd
    if args.loss == 'ce_with_mu_uniform':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss, loss_ce, loss_mu = criterion(penulti_ftrs, labels)
        loss_svd = criterion_svd(penulti_ftrs, labels)
        loss += loss_svd
    if args.loss == 'ce_with_uniform':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss = criterion(penulti_ftrs, labels)
        loss_svd = criterion_svd(penulti_ftrs, labels)
        loss += loss_svd
    if args.loss == 'lb':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss = criterion(penulti_ftrs, labels)
        
    if args.flag_entropy:
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss_entropy = criterion_entropy(penulti_ftrs)
        loss += loss_entropy * args.lambda_loss_var

    if args.flag_entropy_clswise:
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss_entropy = criterion_entropy(penulti_ftrs, labels)
        loss += loss_entropy * args.lambda_loss_var

    if args.loss == 'final':
        outputs, penulti_ftrs = get_features(model, args, inputs)
        loss_ce = criterion(penulti_ftrs, labels)
        loss_reg, (loss_mu, loss_svd, loss_entropy) = criterion_entropy(penulti_ftrs, labels)
        loss = loss_ce + loss_reg
        
    return loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs

def validation(model, criterion, train_loader_mu, val_loader, args):
    with torch.no_grad():
        model.eval()
        val_mu_pos, val_mu_neg = None, None
        if ((args.loss == 'contrastive') | (args.loss == 'triplet')| (args.loss == 'ovadm') | (args.loss == 'arcface')|(args.loss == 'mu_var')| (args.loss == 'mu_var_ver2') | (args.loss == 'mu_var_diag')) & (args.learn_type == 'mu'):
            ftrs_cls0 = []
            ftrs_cls1 = []
            for data in train_loader_mu:
                images, labels = data
                images = images.to('cuda')
                labels = labels.to('cuda')
                outputs, penulti_ftrs = get_features(model, args, images)
                ftrs_cls0.append(penulti_ftrs[labels == 0])
                ftrs_cls1.append(penulti_ftrs[labels == 1])
            # val_mu_pos = torch.cat(ftrs_cls1, dim=0).mean(dim=0)
            # val_mu_neg = torch.cat(ftrs_cls0, dim=0).mean(dim=0)
            val_mu_pos = torch.cat(ftrs_cls0, dim=0).mean(dim=0)
            val_mu_neg = torch.cat(ftrs_cls1, dim=0).mean(dim=0)

        val_total = 0
        val_correct = 0
        for data in val_loader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs, penulti_ftrs = get_features(model, args, images)
            predicted, predicted_value = criterion.predict(penulti_ftrs)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_acc = (100 * val_correct / val_total)
    return val_acc


def train(args, epochs):
    #############################
    # define summary writer
    #############################
    run_dir = "./runs/"
    run_dir = get_run_dir(run_dir, args)
    os.makedirs(run_dir, exist_ok=True)
    summary = SummaryWriter(run_dir)

    ##############################
    # define model/log pth
    ##############################
    model_pth = os.path.join(args.result_path, "models")
    log_pth = os.path.join(args.result_path, "logs")
    os.makedirs(model_pth, exist_ok=True)
    os.makedirs(log_pth, exist_ok=True)

    best_file = os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))
    log_file = os.path.join(args.result_path, "logs", "train_log_{}.txt".format(args.seed))
    with open(log_file, "w") as file:
        file.write("")

    #######################################################
    # define dataset , model, criterion & optimizer
    #######################################################
    train_loader, train_loader_mu, val_loader, _ = get_loaders(output_type= 'single', aug_type = args.aug_type, args = args)
    model = define_model(args)
    if args.loss == 'ce':
        criterion = BaseLoss(args, model.fc[-1])
    elif args.loss == 'ce_with_mu':
        criterion = CEwithMu(args, model.fc[-1])
    elif args.loss == 'ce_with_mu_svd':
        criterion = CEwithMu(args, model.fc[-1])
        criterion_svd = SVD_L(args)
    elif args.loss == 'ce_with_svd':
        criterion = BaseLoss(args, model.fc[-1])
        criterion_svd = SVD_L(args)
    elif args.loss == 'ce_with_mu_uniform':
        criterion = CEwithMu(args, model.fc[-1])
        criterion_svd = Uniform_L(args)
    elif args.loss == 'ce_with_uniform':
        criterion = BaseLoss(args, model.fc[-1])
        criterion_svd = Uniform_L(args)
    elif args.loss == 'lb':
        criterion = LabelSmoothingCrossEntropy(args, model.fc[-1])
    if args.flag_entropy:
        criterion_entropy = HLoss()
    if args.flag_entropy_clswise:
        criterion_entropy = HLoss_classwise()
    if args.loss == 'final':
        criterion = BaseLoss(args, model.fc[-1])
        if args.flag_v2:
            criterion_entropy = CEwithMuVarCov2(args)
        elif args.flag_v3:
            criterion_entropy = CEwithMuVarCov3(args)
        else:
            criterion_entropy = CEwithMuVarCov(args)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    # opt_level = 'O2'
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
        # from apex.parallel import DistributedDataParallel as DDP
        # model=DDP(model,delay_allreduce=True)

    # optimizer = optim.Adam(model.parameters(), lr=3e-4)  # original: 3e-5

    ##############################
    # training
    ##############################
    iteration_for_summary = 0
    best_acc = 0

    for epoch in tqdm(range(epochs)):# original
        model.train()
        running_loss = 0.0
        running_loss_ce, running_loss_mu = 0.0, 0.0
        running_loss_svd, running_loss_entropy = 0.0, 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            iteration_for_summary += 1
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            ########################
            # calc total loss
            ########################
            optimizer.zero_grad()
            if 'svd' in args.loss: 
                loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs = calc_loss(model, criterion, inputs, labels, args, criterion_svd = criterion_svd)
            elif 'uniform' in args.loss:
                loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs = calc_loss(model, criterion,
                                                                                                  inputs, labels, args,
                                                                                                  criterion_svd=criterion_svd)
            elif args.flag_entropy | args.flag_entropy_clswise:
                loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs = calc_loss(model, criterion, inputs, labels, args, criterion_entropy = criterion_entropy)
            elif args.loss == 'final':
                loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs = calc_loss(model, criterion, inputs, labels, args, criterion_entropy = criterion_entropy)
            else:
                loss, loss_ce, loss_mu, loss_svd, loss_entropy, penulti_ftrs, outputs = calc_loss(model, criterion, inputs, labels, args)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if 'ce_with_mu' in args.loss:
                running_loss_ce += loss_ce.item()
                running_loss_mu += loss_mu.item()
            if 'svd' in args.loss:
                running_loss_svd += loss_svd.item()
            if 'uniform' in args.loss:
                running_loss_svd += loss_svd.item()
            if args.flag_entropy | args.flag_entropy_clswise:
                running_loss_entropy += loss_entropy.item()
            if 'final' in args.loss:
                running_loss_ce += loss_ce.item()
                running_loss_mu += loss_mu.item()
                running_loss_svd += loss_svd.item()
                running_loss_entropy += loss_entropy.item()
            #######################
            # prediction & acc
            #######################
            predicted, _ = criterion.predict(penulti_ftrs)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #######################
            # write logs
            #######################
            if (iteration_for_summary != 0) & (iteration_for_summary % 10 == 0):
                table = '[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss / 10)
                summary.add_scalar('loss/loss_tot', (running_loss / 10), iteration_for_summary)
                # summary.add_scalar('loss/loss_ce', (running_loss / 10), iteration_for_summary)

                if args.loss == 'ce_with_mu':
                    summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_mu', (running_loss_mu / 10), iteration_for_summary)
                if 'svd' in args.loss:
                    summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_mu', (running_loss_mu / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_svd', (running_loss_svd / 10), iteration_for_summary)
                if 'uniform' in args.loss:
                    summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_mu', (running_loss_mu / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_svd', (running_loss_svd / 10), iteration_for_summary)
                if args.flag_entropy | args.flag_entropy_clswise:
                    summary.add_scalar('loss/loss_entropy', (running_loss_entropy / 10), iteration_for_summary)
                if args.loss == 'final':
                    summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_mu', (running_loss_mu / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_svd', (running_loss_svd / 10), iteration_for_summary)
                    summary.add_scalar('loss/loss_entropy', (running_loss_entropy / 10), iteration_for_summary)

                train_acc = 100 * (correct / total)
                table += 'Train acc: {}\n'.format(train_acc)
                summary.add_scalar('acc/train_accuracy', train_acc, iteration_for_summary)
                with open(log_file, "a") as file:
                    file.write(table)
                running_loss = 0.0
                if args.loss == 'ce_with_mu':
                    running_loss_ce = 0.0
                    running_loss_mu = 0.0
                if 'svd' in args.loss:
                    running_loss_svd = 0.0
                if 'uniform' in args.loss:
                    running_loss_svd = 0.0
                if args.flag_entropy | args.flag_entropy_clswise:
                    running_loss_entropy = 0.0
                if args.loss == 'final':
                    running_loss_ce = 0.0
                    running_loss_mu = 0.0
                    running_loss_svd = 0.0
                    running_loss_entropy = 0.0

                correct = 0
                total = 0

        #######################
        # validation
        #######################
        val_acc = validation(model, criterion, train_loader_mu, val_loader, args)

        table = 'Epoch: {}, Validation acc: {}'.format(epoch + 1, val_acc)
        summary.add_scalar('acc/val_accuracy', val_acc, epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_file)
            table += "   <<< best acc"

        print(table)
        with open(log_file, "a") as file:
            file.write(table)
            file.write("\n")


def test(args):
    log_file = os.path.join(args.result_path, "logs", "test_log_{}.txt".format(args.seed))
    with open(log_file, "w") as file:
        file.write("")

    _, train_loader_mu, _, test_loader = get_loaders(output_type='single', aug_type=args.aug_type, args=args)

    model = define_model(args)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))

    criterion = BaseLoss(args, model.fc[-1])

    test_acc = validation(model, criterion, train_loader_mu, test_loader, args)
    table = 'Test acc: {}'.format(test_acc)
    with open(log_file, "a") as file:
        file.write(table)

