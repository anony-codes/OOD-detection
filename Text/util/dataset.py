import pandas as pd
from overrides import overrides
from glob import iglob
from sklearn.model_selection import train_test_split
import random
import os
import json
from collections import Counter
# TRAINSIZE=10000
# EVALSIZE=3000
# TESTSIZE=3000
MIN_SIZE=500


class Dataset(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        return NotImplementedError


class ReviewDataset(Dataset):
    def __init__(self,file_path,args,cache_dir,start=2005,end=2014):
        super(ReviewDataset, self).__init__(file_path=file_path)
        self.args=args
        self.all_file=iglob(self.file_path+"/*.json")
        # file=[file for file in self.all_file][0]
        # self.time_series_file=file
        self.cache_dir = cache_dir
        self.start=start
        self.end =end+1

        self.label_map = {1: 0, 2: 0, 4: 1, 5: 1,3:2}

    def _load_json(self,file_path):

        json_data = {i: [] for i in range(1, 6)}
        extracted = {i: [] for i in range(1, 6)}
        label = file_path.split(".")[0]

        with open(file_path,"r") as reader:
            lines=reader.readlines()

        c=Counter()
        naive=[]
        for line in lines:
            line=json.loads(line)
            naive_length=len(line["reviewText"].split(" "))
            naive.append(naive_length)
            if naive_length<self.args.min_lens or naive_length>self.args.max_lens:
                continue
            rating=int(line["overall"])
            json_data[rating].append(line)

        c.update(naive)

        for key,value in json_data.items():
            if key==1:
                sampled = random.sample(json_data[key], TRAINSIZE+EVALSIZE+TESTSIZE)
            else:
                sampled = random.sample(json_data[key],TESTSIZE)
            extracted[key].extend(sampled)

        return extracted,label

    def _load_json_with_time(self, file_path):

        json_data = {}

        with open(file_path, "r") as reader:
            lines = reader.readlines()

        for line in lines:
            line = json.loads(line)
            rating=int(line["overall"])
            if "reviewText" not in line:
                continue

            naive_length = len(line["reviewText"].split(" "))
            if naive_length < self.args.min_lens or naive_length > self.args.max_lens:
                continue

            year = int(line["reviewTime"].replace(",", "").split(" ")[-1])

            if year < self.start:
                continue

            if year not in json_data:
                json_data[year]=[]
                # continue
            line["label"]=int(self.label_map[int(line["overall"])])
            json_data[year].append(line)


        extracted = dict()

        for key, value in json_data.items():

            if len(value)<MIN_SIZE:
                continue

            if key not in extracted:
                extracted[key]=[]

            extracted[key].extend(value)

        return extracted

    def _preprocess(self,dataset,label):

        out = []
        for data in dataset:
            out.append({"text":data["reviewText"],"rating":int(data["overall"]),"label":label})

        return out

    def _preprocess_with_time(self,dataset,category):
        out = []
        neutral=[]
        for data in dataset:
            year=data["reviewTime"].replace(",","").split(" ")[-1]
            if int(data["overall"])!=3:
                out.append({"text":data["reviewText"],"rating":int(data["overall"]),"label":data["label"],"year":year,"category":category})
            else:
                neutral.append(
                    {"text": data["reviewText"], "rating": int(data["overall"]), "label": data["label"], "year": year,
                     "category": category})
        return out,neutral


    def _train_test_split(self,dataset):

        data_size=len(dataset)
        train_dev, test = train_test_split(dataset, test_size=int(data_size*0.1))
        train, dev = train_test_split(train_dev, test_size=int(data_size*0.1))

        return train,dev,test


    @overrides
    def parse(self):
        train_dataset=[]
        dev_dataset=[]
        ood_dataset = {}
        neutral_dataset = {}

        if  self.args.time_series:
            regularized= {}
            neutral_regularized={}
            for file in self.all_file:
                print(file)
                data_category=file.split("/")[-1]
                dataset=self._load_json_with_time(file)

                for key, data in dataset.items():
                    if key not in regularized:
                        regularized[key]=[]
                        neutral_regularized[key]=[]
                    out,neutral=self._preprocess_with_time(data, data_category)
                    regularized[key].extend(out)
                    neutral_regularized[key].extend(neutral)


            for year,examples in regularized.items():
                if year not in ood_dataset:
                    ood_dataset[year]=[]
                if year == self.start:
                    train, dev, test = self._train_test_split(examples)
                    train_dataset.extend(train)
                    dev_dataset.extend(dev)
                    ood_dataset[year].extend(test)
                else:
                    ood_dataset[year].extend(examples)

                print(" dataset {0} has {1} examples".format(year,len(examples)))

            for year,examples in neutral_regularized.items():
                if year not in neutral_dataset:
                    neutral_dataset[year]=[]
                neutral_dataset[year].extend(examples)
                print("neutral dataset {0} has {1} examples".format(year,len(examples)))


        else:
            for file in self.all_file:
                print(file)
                dataset,label=self._load_json(file)

                for key,data in dataset.items():
                    regularized=self._preprocess(data,label)
                    if key==1:
                        train,dev,test=self._train_test_split(regularized)
                        train_dataset.extend(train)
                        dev_dataset.extend(dev)
                        ood_dataset[key].extend(test)
                    else:
                        ood_dataset[key].extend(regularized)

        pd.DataFrame(train_dataset).to_pickle(os.path.join(self.cache_dir,"train.pkl"))
        pd.DataFrame(dev_dataset).to_pickle(os.path.join(self.cache_dir, "dev.pkl"))

        for key,value in ood_dataset.items():
            pd.DataFrame(value).to_pickle(os.path.join(self.cache_dir, "ood_{0}.pkl".format(key)))

        for key,value in neutral_dataset.items():
            pd.DataFrame(value).to_pickle(os.path.join(self.cache_dir, "neutral_{0}.pkl".format(key)))


