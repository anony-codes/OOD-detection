import pickle
import argparse

from utils.trainer import *
from model.model_utils import *

from utils.odin import sample_odin_estimator, get_odin_score

features = None
            
def test(object, args, train_loader = None, loaders = None, val_loader = None, device = None):
    if (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
        model = object.model
    else:
        model = object
        criterion = BaseLoss(args, model.fc[2])
    total = 0
    correct = 0
    dict_results = dict()
    stat_results = dict()
#     dict_results['data'] = []
    dict_results['preds'] = []
    dict_results['trues'] = []
    dict_results['correct'] = []
    dict_results['dataset_idx'] = []
    dict_results['org_labels'] = []
    dict_results['pred_labels'] = []
    if ((args.loss == 'ce')|(args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd')| (args.loss == 'ce_with_svd') ) & (args.metric == 'baseline'):
        dict_results['checksum'] = []
        dict_results['full_preds'] = []
    if args.loss == 'lb':
        dict_results['full_preds'] = []

    test_loaders = loaders
    print(len(test_loaders))
    
#     if args.metric == 'mahalanobis':
    feature_list = get_multiple_feature_list(model, args, device=args.device)

    if 'mahalanobis' in args.metric:
        if (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
            mean, var = sample_estimator(model, args, num_classes=args.num_classes, feature_list= feature_list, train_loader = train_loader,
                                     device=args.device, pred_type=args.pred_type_mahala, flag_simclr = True)
        else:
            mean, var = sample_estimator(model, args, num_classes=args.num_classes, feature_list=feature_list,
                                         train_loader=train_loader,
                                         device=args.device, pred_type=args.pred_type_mahala, flag_simclr=False)

    else:
        mean, var = None, None

    if 'odin' in args.metric:
        odin_criterion = nn.CrossEntropyLoss()
        epsilon = sample_odin_estimator(model, odin_criterion, val_loader, epsilons=[0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])

    stat_results['mean'] = mean
    stat_results['var'] = var

    for idx, loader in enumerate(test_loaders):
        value_ind = None
        value_ood = None
        for idx_ins, data in tqdm(enumerate(loader)):
            images, labels = data

            if idx == 0:
                labels_ood = torch.ones(labels.size(0))
            else:
                labels_ood = torch.zeros(labels.size(0))
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels_ood = labels_ood.to(args.device)

            #get outputs
            if (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
                predicted, predicted_value = object.predict(images)
            else:
                outputs, penulti_ftrs = get_features(model, args, images)
                predicted, predicted_value = criterion.predict(penulti_ftrs)

            #get ood statistics
            if (args.metric == 'baseline'):
                if (args.loss =='ce') | (args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd') | (args.loss == 'ce_with_svd') | (args.loss == 'lb'):
                    dict_results['full_preds'] += (predicted_value).tolist()
                    predicted_ood, _ = torch.max(predicted_value, dim=-1)
                else:
                    predicted_ood = predicted_value
            elif (args.metric == 'cos_similarity') | (args.metric == 'cos_similarity_mu') | (args.metric == 'baseline_mu'):
                predicted_ood = predicted_value
            elif 'mahalanobis' in args.metric:
                predicted_ood = get_ood_outputs(model, images, args, stat_mahala = (mean, var), device = args.device, pred_type =
                args.pred_type_mahala)
            elif 'odin' in args.metric:
                predicted_ood = get_odin_score(args,model, images, temperature=1000, epsilon = epsilon, criterion = odin_criterion )
            else:
                print('metric is not available')

            if idx == 0:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

#             dict_results['data']+=(images).tolist()
            dict_results['preds']+=(predicted_ood).tolist()
            dict_results['trues']+=(labels_ood).tolist()
            dict_results['org_labels']+=(labels).tolist()
            dict_results['pred_labels']+=(predicted).tolist()
            if (args.loss == 'ce') & (args.metric == 'baseline'):
                dict_results['checksum'] += (torch.sum(images.flatten(1), dim = 1)).tolist()
            if idx ==0:
                dict_results['correct']+=(predicted == labels).tolist()
            else:
                dict_results['correct']+=(torch.zeros_like(labels)-1).tolist()
            dict_results['dataset_idx']+=(torch.zeros_like(labels)+idx).tolist()

            if idx == 0:
                if value_ind is None:
                    value_ind = predicted_ood
                else:
                    value_ind = torch.cat([value_ind, predicted_ood])
            else:
                if value_ood is None:
                    value_ood = predicted_ood
                else:
                    value_ood = torch.cat([value_ood,predicted_ood])


    print(correct)
    print(total)
    acc = (100 * correct / total)
    dict_results['acc'] = acc
    print('Accuracy of the network on the test images: {} %%'.format(acc))
    

    if args.flag_adjust:
        dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed))
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "scores_adjust_{}.pkl".format(args.type_adjust))
    else:
        if ((args.loss == 'contrastive') | (args.loss == 'triplet')) & (args.metric == 'baseline'):
            dir_path = os.path.join(args.result_path, args.metric, args.pred_type, "seed_" + str(args.seed))
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(args.result_path, args.metric, args.pred_type, "seed_" + str(args.seed), "scores_age.pkl")
        elif ((args.loss == 'contrastive') | (args.loss == 'triplet') | (args.loss == 'ce_with_mu')) & (args.metric == 'mahalanobis'):
            dir_path = os.path.join(args.result_path, args.metric, args.pred_type_mahala, "seed_" + str(args.seed))
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(args.result_path, args.metric, args.pred_type_mahala, "seed_" + str(args.seed), "scores_age.pkl")
        else:
            dir_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed))
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(args.result_path, args.metric, "seed_" + str(args.seed), "scores_age.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(dict_results, f)

    if 'mahalanobis' in args.metric:
        stat_path = os.path.join(dir_path, "mean_var.pkl")
        with open(stat_path, "wb") as f:
            pickle.dump(stat_results, f)
        
    
  
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, help='baseline mahala')
    parser.add_argument('--flag_adjust', action='store_true', help='adjust test or not')
    parser.add_argument('--type_adjust', type=str, help='path of the model')
    parser.add_argument('--num_classes', default = 2, type=int, help='path of the model')
    parser.add_argument('--result_path', default="./results", type=str, help='train or test')
    parser.add_argument('--loss', default="ce", type=str, help='train or test')
    parser.add_argument('--seed', default = 0, type=int, help='path of the model')
    parser.add_argument('--epochs', default=100, type=int, help='number of classes')

    parser.add_argument('--M', default = 1., type=float, help='path of the model')
    parser.add_argument('--flag_margin', action='store_true', help='get margin or not')

    parser.add_argument('--M_var', default=1., type=float, help='path of the model')
    parser.add_argument('--S', default = 1., type=float, help='path of the model')
    parser.add_argument('--learn_type', default="mu", type=str, help='w or mu')
    parser.add_argument('--pred_type', default="softmax", type=str, help='dist or normalized or softmax')
    parser.add_argument('--pred_type_mahala', default="org", type=str, help='org or diagonal or l2')
    parser.add_argument('--exp_ratio', default=1., type=float, help='Margin')
    parser.add_argument('--aug_type', default="basic", type=str, help='simclr or basic')
    parser.add_argument('--batch_size', default=128, type=int, help='number of classes')
    parser.add_argument('--smooth_factor', default=1., type=float, help='Sphere size')
    parser.add_argument('--lambda_loss_mu', default=1., type=float, help='lambda loss mu')
    parser.add_argument('--temperature', default=2., type=float, help='lambda loss mu')
    parser.add_argument('--lambda_loss_var', default=1., type=float, help='lambda loss mu')
    parser.add_argument('--lambda_loss_corr', default=1., type=float, help='lambda loss mu')
    
    parser.add_argument('--flag_entropy', action='store_true', help='get margin or not')
    parser.add_argument('--flag_entropy_clswise', action='store_true', help='get margin or not')
    parser.add_argument('--flag_v2', action='store_true', help='get margin or not')
    parser.add_argument('--flag_v3', action='store_true', help='get margin or not')
    
    args = parser.parse_args()


    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    train_dataset = UTKDataset_Simclr(mode = 'train', output_type='single', aug_type='none') # no data augmentation
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    _, _, val_loader, _ = get_loaders(output_type='single', aug_type=args.aug_type, args=args)

    ##############################
    # Model
    ##############################
    if (args.loss == 'ce')| (args.loss == 'ce_with_mu') | (args.loss == 'ce_with_mu_svd') | (args.loss == 'ce_with_svd') | (args.loss == 'lb') | ('uniform' in args.loss) | (args.loss == 'final'):
        model = define_model(args)
        model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
        model.eval()
        test(model, args, train_loader=train_loader, loaders=loaders, val_loader = val_loader, device=args.device)
    elif (args.loss == 'ce_simclr') | (args.loss == 'ce_simclr_negative')| (args.loss == 'ce_simclr_with_mu')| (args.loss == 'ce_simclr_negative_with_mu'):
        simclr = define_model(args)
        simclr.model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
        simclr.model.eval()

        test(simclr, args, train_loader=train_loader, loaders=loaders, val_loader = val_loader, device=args.device)
    else:
        raise NotImplementedError

            
            
if __name__ == '__main__':
    main()