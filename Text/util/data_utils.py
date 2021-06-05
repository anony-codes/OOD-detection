from .dataset import ReviewDataset
import os
from .preprocessor import Processor,ContrastiveProcessor
import pandas as pd
import logging
import glob
from dataclasses import dataclass

logger=logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TRAINSIZE=24000
DEVSIZE=3000
OODSIZE=3000



def get_review_dataset(args,tokenizer,is_test:bool=False):
    data_dir = os.path.join(args.root,args.dataset)
    post_fix = "time" if args.time_series else "category"
    cache_dir = os.path.join(data_dir, "cache_{0}".format(post_fix))

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    cache_files=glob.iglob(cache_dir+"/*.pkl")
    cache_files= [file for file in cache_files]

    if len(cache_files)<4:
        logger.info("Sample data from Raw in %s"%(str(cache_dir)) )
        review_dataset=ReviewDataset(data_dir,args,cache_dir)
        review_dataset.parse()

    dataset_dir=os.path.join(cache_dir,args.encoder_class)

    if args.use_custom_vocab:
        dataset_dir +="-custom"

    indexed_files = glob.iglob(os.path.join(dataset_dir + "/*.indexed"))
    indexed_files=[i for i in indexed_files]

    if len(indexed_files) < 4:

        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        cache_files=glob.iglob(cache_dir+"/*.pkl")
        cache_files=sorted([f for f in cache_files],reverse=False)
        processor=Processor(tokenizer,args)
        for cache_file in cache_files:
            f_path=cache_file.split("/")[-1]
            neutral= True if "neutral" in f_path else False
            if "train.pkl" in f_path:
                size=TRAINSIZE
            elif "dev.pkl" in f_path:
                size=DEVSIZE
            else:
                size=OODSIZE
            df=processor.preprocess(cache_file,number=size,neutral=neutral)
            df.to_pickle(os.path.join(dataset_dir,f_path+".indexed"))

    oods= {"ood":[]}
    indexed_files = glob.iglob(os.path.join(dataset_dir + "/*.indexed"))
    indexed_files=[i for i in indexed_files]
    ood_id=[]

    for indexed_file in indexed_files:
        if "neutral" in indexed_file:
            continue
        print(indexed_file)
        if "train" in indexed_file:
            train=pd.read_pickle(indexed_file)
        elif "dev" in indexed_file:
            dev = pd.read_pickle(indexed_file)
        else:
            if "2005" in indexed_file:
                oods["test"]=pd.read_pickle(indexed_file)

            oods["ood"].append(pd.read_pickle(indexed_file))
            ood_id.append(indexed_file.split("/")[-1])

    return train, dev, oods, ood_id



def get_contrastive_dataset(args,tokenizer):

    data_dir = os.path.join(args.root,args.dataset)
    post_fix = "time" if args.time_series else "category"
    cache_dir = os.path.join(data_dir, "cache_{0}".format(post_fix))


    processor = ContrastiveProcessor(tokenizer, args)

    contrastive_path="data/contrastive/train_contrastive.pkl"

    if os.path.isfile(contrastive_path):
        train_contrastive=pd.read_pickle(contrastive_path)

    else:

        train_contrastive = processor.preprocess("data/contrastive/augment.pkl")
        pd.to_pickle(train_contrastive,contrastive_path)

    return train_contrastive