import argparse
from utils.trainer import *
from model.model_utils import *
from utils.mahalanobis import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
features = None


def get_features_from_loader(model, loader, args):
    features = []
    targets = []
    with torch.no_grad():
        for idx_ins, data in tqdm(enumerate(loader)):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)

            _, feautre_small = get_features(model, args, images)  # N, 128
            features.append(feautre_small)
            targets.append(labels)
        features_all = torch.cat(features)
        trgs_all = torch.cat(targets)
    return features_all, trgs_all


def get_trainftrs(model, args, train_loader = None):
    model = model.to(args.device)
    model.eval()

    features_all, trgs_all = get_features_from_loader(model, train_loader, args)

    dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "penultimate_ftrs(train)")
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "ftrs_age_{}.npy".format('train'))
    trg_pth = os.path.join(dir_path, "trgs_age_{}.npy".format('train'))

    np.save(file_path, features_all.detach().cpu().numpy())
    np.save(trg_pth, trgs_all.detach().cpu().numpy())

            
def test(model, args, loaders = None):
    model = model.to(args.device)
    model.eval()

    test_loaders = loaders
    print(len(test_loaders))

    for idx, loader in enumerate(test_loaders):
        ###############
        # get features from the loader
        ###############
        features_all, trgs_all = get_features_from_loader(model, loader, args)

        dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "penultimate_ftrs")
        os.makedirs(dir_path, exist_ok=True)
        if args.flag_adjust:
            file_path = os.path.join(dir_path, "ftrs__{}_{}.npy".format(args.type_adjust, idx))
            trg_pth = os.path.join(dir_path, "trgs_{}_{}.npy".format(args.type_adjust, idx))
        else:
            file_path = os.path.join(dir_path, "ftrs_age_{}.npy".format(idx))
            trg_pth = os.path.join(dir_path, "trgs_age_{}.npy".format(idx))

        np.save(file_path, features_all.detach().cpu().numpy())
        np.save(trg_pth, trgs_all.detach().cpu().numpy())


            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default = 'mahalanobis', type=str, help='mahalanobis or baseline')
    parser.add_argument('--flag_adjust', action='store_true', help='adjust test or not')
    parser.add_argument('--type_adjust', type=str, help='path of the model')
    parser.add_argument('--num_classes', default = 2, type=int, help='path of the model')
    parser.add_argument('--result_path', default="./results", type=str, help='train or test')
    parser.add_argument('--seed', default = 0, type=int, help='path of the model')
    parser.add_argument('--M', default=1., type=float, help='Margin')
    parser.add_argument('--M_var', default=1., type=float, help='Margin')
    parser.add_argument('--S', default=1., type=float, help='Margin')
    parser.add_argument('--loss', default="ce", type=str, help='ce or triplet or arcface')
    parser.add_argument('--contrastive_full', action='store_true', help='contrastive full ver.' )
    parser.add_argument('--exp_ratio', default=1., type=float, help='Margin')
    parser.add_argument('--learn_type', default="mu", type=str, help='w or mu')
    parser.add_argument('--aug_type', default="mu", type=str, help='w or mu')
    parser.add_argument('--flag_margin', action='store_true', help='get margin or not')
    parser.add_argument('--flag_entropy', action='store_true', help='get margin or not')
    parser.add_argument('--flag_entropy_clswise', action='store_true', help='get margin or not')
    parser.add_argument('--smooth_factor', default=1., type=float, help='Sphere size')
    parser.add_argument('--lambda_loss_mu', default=1., type=float, help='lambda loss mu')
    parser.add_argument('--lambda_loss_var', default=1., type=float, help='lambda loss mu')
    parser.add_argument('--lambda_loss_corr', default=1., type=float, help='lambda loss mu')
    parser.add_argument('--temperature', default=2., type=float, help='lambda loss mu')

    parser.add_argument('--epochs', default=100, type=int, help='path of the model')
    parser.add_argument('--batch_size', default=128, type=int, help='number of classes')
    parser.add_argument('--flag_v2', action='store_true', help='get margin or not')
    parser.add_argument('--flag_v3', action='store_true', help='get margin or not')

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.result_path = get_result_path(args)
    print(args)

    set_seed(args.seed)

    
    ##############################
    # Data
    ##############################
    if args.flag_adjust:
        loaders, data_len, _ = get_adjust_dataloaders(args.type_adjust)
    else:
        loaders, data_len = get_eval_200_dataloaders()
    train_dataset = UTKDataset_Simclr(mode = 'train', output_type='single', aug_type='none')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    ##############################
    # Model
    ##############################
    # model = define_model(args)
    # model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    # model.eval()

    if (args.loss == 'ce') | (args.loss == 'ce_with_mu')|(args.loss=='lb')| (args.loss == 'ce_with_mu_svd')| (args.loss == 'ce_with_svd')  | (args.loss == 'final'):
        model = define_model(args)
        model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
        model.eval()
        test(model, args, loaders=loaders)
        get_trainftrs(model, args, train_loader=train_loader)
    elif (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
        simclr = define_model(args)
        simclr.model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
        simclr.model.eval()
        test(simclr.model, args, loaders=loaders)
        get_trainftrs(simclr.model, args, train_loader=train_loader)

    ###############################
    # Test
    ###############################
    # test(model, args, loaders = loaders)
    # get_trainftrs(model, args,train_loader = train_loader)
            
if __name__ == '__main__':
    main()