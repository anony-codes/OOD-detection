import os
import argparse


class ExperimentArgument:
    def __init__(self):

        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):

        parser = argparse.ArgumentParser(description="OOD analysis")
        parser.add_argument("--root", type=str, required=True)
        parser.add_argument("--seed", type=int, default=777)
        parser.add_argument("--dataset", choices=["product","contrastive"], type=str, required=True)
        parser.add_argument("--model", choices=["transformer"], type=str,
                            default="transformer")
        parser.add_argument("--repre", choices=["term", "tfidf"], type=str, default="term")

        parser.add_argument("--encoder_class", choices=["bert-base-uncased", "bert-large-uncased", "distilbert","prajjwal1/bert-small"],
                            required=True,
                            type=str)
        parser.add_argument("--metric",
                            choices=["basic","regularizer","lb","ssd","contrastive"],
                            default="basic", type=str)

        parser.add_argument("--learn_type", choices=["w", "mu"], default="w",
                            type=str)  # just for comparison btw contrastive , triplet and ovadm
        parser.add_argument("--return_type", choices=["prob", "distance"], default="prob", type=str)

        parser.add_argument("--min_lens", default=10, type=int)
        parser.add_argument("--max_lens", default=300, type=int)

        parser.add_argument("--lr", default=3e-6, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--warmup_step", default=0, type=int)
        parser.add_argument("--decay_step", default=20000, type=int)
        parser.add_argument("--clip_norm", default=0.25, type=float)
        parser.add_argument("--n_epoch", default=5, type=int)
        parser.add_argument("--lambda_mu", default=0.1, type=float)
        parser.add_argument("--lambda_var", default=10.0, type=float)
        parser.add_argument("--lambda_corr", default=1.0, type=float)

        parser.add_argument("--scale", default=0.3, type=float)
        parser.add_argument("--alpha", default=0.5, type=float)

        parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
        parser.add_argument("--vis_dir", default="runs", type=str)

        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--regenerate_ood", action="store_true")
        parser.add_argument("--save_weight", action="store_true")
        parser.add_argument("--ensemble_test", action="store_true")

        parser.add_argument("--use_custom_vocab", action="store_true")  # should be used in scratch model
        parser.add_argument("--time_series", action="store_true")
        parser.add_argument("--include_neutral", action="store_true")
        parser.add_argument("--lower_dim", action="store_true")
        parser.add_argument("--pool", action="store_true")
        parser.add_argument("--ensemble", action="store_true")

        parser.add_argument("--scratch", action="store_true")
        parser.add_argument("--model_path", default="", type=str)
        parser.add_argument("--save_example_path", default="", type=str)
        parser.add_argument("--ood_rate", default=0.1, type=float)
        parser.add_argument("--english_only", action="store_true")
        parser.add_argument("--layer_id", type=int, default=2)

        return parser

    def set_savename(self):

        if self.data["use_custom_vocab"] and not self.data["scratch"]:
            raise ValueError


        if self.data["metric"] == "basic":
            self.data["savename"] = self.data["model"] + "-{0}-{1}".format(
                self.data["seed"],
                self.data["metric"])

        elif self.data["metric"] == "regularizer":
            self.data["savename"] = self.data["model"] + "-{0}-{1}-{2}-{3}-{4}".format(self.data["seed"],
                                                                               self.data["metric"],
                                                                               self.data["lambda_mu"],
                                                                               self.data["lambda_var"],
                                                                               self.data["lambda_corr"])

        # elif self.data["metric"] == "ssd":
        #     self.data["savename"] = self.data["model"] + "-{0}-{1}".format(self.data["seed"],
        #                                                                        self.data["metric"])
        # elif self.data["metric"] == "contrastive":
        #     self.data["savename"] = self.data["model"] + "-{0}-{1}".format(self.data["seed"],
        #                                                                        self.data["metric"])


        if self.data["use_custom_vocab"]:
            self.data["savename"] += "-custom"

        self.data["checkpoint_name"] = os.path.join(self.data["checkpoint_dir"], self.data["savename"])
        if not os.path.isdir(self.data["checkpoint_name"]):
            os.makedirs(self.data["checkpoint_name"])

        if self.data["do_test"]:
            self.data["save_example_path"] = os.path.join(self.data["save_example_path"], self.data["model"])
            if not os.path.isdir(self.data["save_example_path"]):
                os.makedirs(self.data["save_example_path"])





class TranslationArgument:
    def __init__(self):

        data = {}
        parser = self.get_args()
        args = parser.parse_args()
        data.update(vars(args))
        self.data = data
        self.__dict__ = data

    def get_args(self):

        parser = argparse.ArgumentParser(description="Augmentations")
        parser.add_argument("--lr", default=3e-6, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        parser.add_argument("--root", type=str, default="./")
        parser.add_argument("--distributed_inference", action="store_true")
        parser.add_argument("--local_rank", type=int)

        return parser

