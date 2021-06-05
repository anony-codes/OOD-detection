import pickle

from model.model_utils import *
from utils.trainer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
features = None


def calc_ratio(in_feature, label, args):
    trg = label[:, None]
    n_sample = in_feature.shape[0]

    mu_matrix = torch.cat([torch.mean(in_feature[label == l], dim=0).unsqueeze(0) \
                           for l in range(0, args.num_classes)], dim=0)
    l2_distance = torch.cdist(in_feature, mu_matrix)

    one_hot_target = (trg == torch.arange(args.num_classes).to(l2_distance.device).reshape(1, args.num_classes)).bool()  # all false distance index
    term1 = l2_distance.gather(dim=1, index=label.unsqueeze(1))  # true distance
    term2 = l2_distance[~one_hot_target].view([n_sample, -1])  # False distance

    pos_mean = torch.mean(term1)
    neg_mean = torch.mean(term2)
    mu_ratio_mean = torch.mean(term2 / term1)
    return pos_mean, neg_mean, mu_ratio_mean


def get_features_from_loader(model, loader, args):
    features = []
    targets = []
    with torch.no_grad():
        for idx_ins, data in tqdm(enumerate(loader)):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            _, feautre_small = get_features(model, images, args.num_classes)
            features.append(feautre_small)
            targets.append(labels)
        features_all = torch.cat(features)
        trgs_all = torch.cat(targets)
    return features_all, trgs_all


def get_ratio(model, args, train_loader=None):
    stat = dict()
    model = model.to(args.device)
    model.eval()
    features_all, trgs_all = get_features_from_loader(model, train_loader, args)
    pos_mean, neg_mean, mu_ratio_mean = calc_ratio(features_all, trgs_all, args)
    stat['pos_mean'] = [pos_mean.item()]
    stat['neg_mean'] = [neg_mean.item()]
    stat['mu_ratio_mean'] = [mu_ratio_mean.item()]

    stat_path = os.path.join(args.result_path, "stats", "seed_" + str(args.seed))
    os.makedirs(stat_path, exist_ok=True)
    with open(stat_path+"/pos_neg_ratio.pkl", "wb") as f:
        pickle.dump(stat, f)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=2, type=int, help='path of the model')
    parser.add_argument('--result_path', default="./results", type=str, help='train or test')
    parser.add_argument('--seed', default=0, type=int, help='path of the model')
    parser.add_argument('--M', default=1., type=float, help='Margin')
    parser.add_argument('--loss', default="ce", type=str, help='ce or triplet or arcface')
    parser.add_argument('--learn_type', default="mu", type=str, help='w or mu')
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.result_path = get_result_path(args)
    print(args)

    set_seed(args.seed)

    ##############################
    # Data
    ##############################
    train_dataset = UTKDataset(mode='train', mahala=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    ##############################
    # Model
    ##############################
    model = define_model(args)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    model.eval()

    ###############################
    # Test
    ###############################
    get_ratio(model, args, train_loader=train_loader)


if __name__ == '__main__':
    main()