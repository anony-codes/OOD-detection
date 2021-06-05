import argparse
from utils.trainer import *

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--run_mode', default="train", type=str, help='train or test')
parser.add_argument('--result_path', default="./results", type=str, help='train or test')
parser.add_argument('--loss', default="ce", type=str, help='ce or ovadm or contrastive or contrastive_mean')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--epochs', default=100, type=int, help='number of classes')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--M', default=1., type=float, help='Margin')
parser.add_argument('--M_var', default=1., type=float, help='Margin')
parser.add_argument('--S', default=1., type=float, help='Sphere size')
parser.add_argument('--learn_type', default="mu", type=str, help='w or mu')
parser.add_argument('--pred_type', default="dist", type=str, help='dist or normalized or softmax')
parser.add_argument('--aug_type', default="basic", type=str, help='simclr or basic')
parser.add_argument('--batch_size', default=128, type=int, help='number of classes')
parser.add_argument('--flag_margin', action='store_true', help='get margin or not')
parser.add_argument('--flag_entropy', action='store_true', help='get margin or not')
parser.add_argument('--flag_entropy_clswise', action='store_true', help='get margin or not')
parser.add_argument('--smooth_factor', default=1., type=float, help='Sphere size')
parser.add_argument('--lambda_loss_mu', default=1., type=float, help='lambda loss mu')
parser.add_argument('--lambda_loss_var', default=1., type=float, help='lambda loss mu')
parser.add_argument('--lambda_loss_corr', default=1., type=float, help='lambda loss mu')
parser.add_argument('--temperature', default=2., type=float, help='lambda loss mu')
parser.add_argument('--flag_v2', action='store_true', help='get margin or not')
parser.add_argument('--flag_v3', action='store_true', help='get margin or not')



parser.set_defaults(argument=True)

def main():
    args = parser.parse_args()
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.result_path = get_result_path(args)

    if args.run_mode == 'train':
        print(args)
        if 'final' in args.loss:
            train(args, epochs=args.epochs)
            test(args)
        elif args.loss == 'ce':
            train(args, epochs=args.epochs)
            test(args)
        else:
            raise NotImplementedError
    else:
        print("not available mode")

if __name__ == '__main__':

    main()
