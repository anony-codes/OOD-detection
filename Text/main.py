import sys

sys.path.insert(0, "../")
# import mahalanobis

from util.odin import sample_odin_estimator
from util.data_utils import *
from util.args import ExperimentArgument
from transformers import AutoTokenizer, RobertaTokenizer, AdamW, BertTokenizer
import apex
import glob
from tqdm import tqdm

# from util.slack import slackBot,slackalarm
from util.batch_generator import *
import os
from util.logger import log_full_eval_test_results_to_file

from model.simclr import NTXentLoss
from model.losses import *
from util.bpe_tokenizer import CustomTokenizer
from pytorch_transformers import WarmupLinearSchedule
from model.transformer import BaseLine
import torch.nn as nn
import random
import logging
from util.trainer import CFTrainer
from util.mahalanobis import *

logging.basicConfig(level=logging.WARN)


def set_seed(args, random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.do_test:
        torch.backends.cudnn.enabled = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_trainer(args, model, train_batchfier, test_batchfier):
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    # optimizer=RAdam(model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    # if args.model == "transformer":
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.mixed_precision and args.do_train:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, args.decay_step)

    if args.metric == "basic":
        criteria = BaseLoss(args, model.classifier)

    elif args.metric=="regularizer":
        criteria = Regularizer(args, model.classifier, args.lambda_mu, lambda_var=args.lambda_var,
                                     lambda_corr=args.lambda_corr)
    else:
        raise NotImplementedError

    trainer = CFTrainer(args, model, train_batchfier, test_batchfier, optimizer, args.gradient_accumulation_step,
                        criteria, args.clip_norm, \
                        args.mixed_precision, args.n_label)

    return trainer


def get_batchfier(args, tokenizer):
    n_gpu = torch.cuda.device_count()
    ood_batch = []
    train, dev, oods, ood_ids = get_review_dataset(args, tokenizer)

    if args.use_custom_vocab:
        padding_idx = tokenizer.token_to_id("[PAD]")

    else:
        padding_idx = tokenizer.pad_token_id

    # if args.model == "transformer":

    train_batch = CFBatchGenerator(train, args.per_gpu_train_batch_size * n_gpu, padding_index=padding_idx)
    dev_batch = CFBatchGenerator(dev, args.per_gpu_eval_batch_size * n_gpu, padding_index=padding_idx)
    test_batch = CFBatchGenerator(oods["test"], args.per_gpu_eval_batch_size * n_gpu, padding_index=padding_idx)

    for ood, ood_id in zip(oods["ood"], ood_ids):
        ood_batch.append(CFBatchGenerator(ood, args.per_gpu_eval_batch_size * n_gpu, padding_index=padding_idx,
                                          dataset_type=ood_id))

    return train_batch, dev_batch, test_batch, ood_batch


def run():
    args = ExperimentArgument()
    args.gpu = "cuda" if torch.cuda.is_available() else "cpu"

    args_description = 'model : {}, loss_func : {}, scratch : {}, lambda : {}'.format(
        args.model, args.metric, args.scratch, args.lambda_mu)
    set_seed(args, args.seed)

    vocab_size = AutoTokenizer.from_pretrained(args.encoder_class).vocab_size
    data_dir = os.path.join(args.root, args.dataset)
    post_fix = "time" if args.time_series else "category"
    cache_dir = os.path.join(data_dir, "cache_{0}".format(post_fix))

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    tokenizer = CustomTokenizer.load_encoder(cache_dir, vocab_size)
    args.vocab_size = vocab_size
    train_gen, dev_gen, test_gen, ood_gens = get_batchfier(args, tokenizer)

    model = BaseLine(args, n_class=2)
    model.to(args.gpu)
    args.n_label = 2

    trainer = get_trainer(args, model, train_gen, dev_gen)
    results = []
    optimal_acc = 0.0
    best_dir = os.path.join(args.checkpoint_name, "best_model")

    print("Save at {}".format(best_dir))
    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    pbar = tqdm(range(args.n_epoch), desc="Epoch: ")

    if args.do_train:
        print(args_description)
        # slackBot.post_message(args_description)
        description = os.path.join(args.checkpoint_name, "performance_per_epoch.txt")
        f = open(description, "w")

        for e in pbar:
            trainer.train_epoch()
            save_path = os.path.join(args.checkpoint_name, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(args, os.path.join(save_path, "training_args.txt"))

            if args.evaluate_during_training:
                accuracy, macro_f1 = trainer.test_epoch()
                results.append({"eval_acc": accuracy, "eval_f1": macro_f1})

                if optimal_acc < accuracy:
                    optimal_acc = accuracy
                    if args.model == "tree":
                        importance = trainer.save_feature_importance()
                        pd.to_pickle(importance, os.path.join(save_path, "importance.bin"))
                    else:
                        torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                        print("Update Model checkpoints at {0}!! ".format(best_dir))

            f.write("Epoch {0} : {1}\n".format(e, accuracy))
        f.close()

    if args.do_eval:
        accuracy, macro_f1 = trainer.test_epoch()
        descriptions = os.path.join(args.checkpoint_name, "eval_results.txt")
        writer = open(descriptions, "w")
        writer.write("accuracy: {0:.4f}, macro f1 : {1:.4f}".format(accuracy, macro_f1) + "\n")
        writer.close()

    if args.do_test:  # for measuring distance and output
        print('return_type : {}'.format(args.return_type))
        print(args_description)

        if args.ensemble_test:
            model_paths = glob.iglob(args.model_path)

        if args.model_path == "":
            raise EnvironmentError("require to clarify the argment of model_path")

        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)
        accuracy, macro_f1 = trainer.test_epoch()


        model.eval()
        sample_dir = os.path.join(args.save_example_path, "{0}".format(args.savename))

        print("Save sample at {0}".format(sample_dir))

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        train, oods = train_gen, ood_gens
        print("in distribution year dataset: {0}".format(train.dataset_type))
        feature_list = get_feature_list(model, flag_synthetic=True, device="cuda")
        statsf = os.path.join(sample_dir, "train_statistics")

        print("Generate train statistics for mahalanobis distance")
        mean, prec, var = sample_estimator(model, num_classes=2, feature_list=feature_list,
                                           train_loader=train, device="cuda", args=args)
        eps = 0.08
        test_dict = {}

        for idx, ood in enumerate(oods):
            print(ood.dataset_type)
            trainer.test_batchfier = ood  # replace with ood loader
            ood_file = trainer.ood_epoch(ood, mean, prec, var, eps, args.n_label)

            test_dict[ood.dataset_type] = ood_file
            class_weight = model.classifier.weight.tolist()

            print("Save OOD dataset results.")
            out = {"dataset": ood_file, "class_weight": class_weight}
            pd.to_pickle(out, os.path.join(sample_dir, "{0}.result.{1}".format(ood.dataset_type, args.return_type)))

        print("Save train results")
        trainer.test_batchifer = train_gen
        train_file = trainer.ood_epoch(train_gen, mean, prec, var, args.n_label)

        train_out = {"dataset": train_file, "class_weight": class_weight}
        pd.to_pickle(train_out, os.path.join(sample_dir, "train.pkl.indexed.result.{0}".format(args.return_type)))

        from util.metrics import auroc
        in_df = test_dict["ood_2005.pkl.indexed"]
        year_sorted = sorted(list(test_dict.keys()))
        auroc_list = {}


        print("compute auroc")
        for m in ["m_distance", "argmax_prob", "odin_score","m_distance_pen"]:
            auroc_list[m] = []
            for year in year_sorted:
                ood_df = test_dict[year]
                auroc_list[m].append(auroc(in_df, ood_df, m))
        print("finish auroc")
        print(year_sorted)
        print(auroc_list)


        pd.to_pickle(auroc_list, os.path.join(sample_dir, "auroc.pkl.indexed.result"))

if __name__ == "__main__":
    run()
