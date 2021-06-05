import pandas as pd
from transformers import BertTokenizer
import random
from tqdm import tqdm
from typing import List, Optional, Union
import json
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
import numpy as np
from tokenizers import BertWordPieceTokenizer
# import pickle
# import pickle5 as pickle

@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass
class MaskAugmentedExample:
    """
        guid: Unique id for the example.
        text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """
    text: np.array # 1d array
    mask_prob:float
    mask_id: int
    label: int

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


    def aug_with_mask(self):
        self.aug_text_mask=deepcopy(self.text)
        mask_lens=int(len(self.aug_text_mask)*self.mask_prob)
        ids_for_mask=random.sample(list(range(0,len(self.aug_text_mask))),k=mask_lens)
        self.aug_text_mask[ids_for_mask]= self.mask_id

    def aug_with_permute(self):
        self.aug_text_perm=deepcopy(self.text)
        np.random.shuffle(self.aug_text_perm)



class Processor(object):
    def __init__(self,tokenizer:BertTokenizer,args):
        self.tokenizer=tokenizer
        self.label_map=None
        self.args=args
    def _load_pickle(self,file_path):
        """

        :param file_path:
        :return: pd.DataFrame
        """
        # with open(file_path, 'rb') as f:
        #     df = pickle.load(f)

        # return df
        return pd.read_pickle(file_path)


    def sample(self,df:pd.DataFrame,n):
        print(df.shape)

        if df.shape[0]<n:
            n=df.shape[0]

        return df.sample(n=n).reset_index(drop=True)


    def preprocess(self,file_path:str,number,neutral):
        dataset=self._load_pickle(file_path)
        if not neutral:
            dataset=self._balance_with_label(dataset)
        dataset=self.sample(dataset,number)

        print("====label distribution====")

        for key,value in dict(dataset.label.value_counts()).items():
            print("label {0} : {1}".format(key,value))

        if self.label_map is None:
            self.label_map={label:idx for idx,label in enumerate(dataset.label.value_counts().keys())}
            self.label_map[3]=2

        df={"text":[],"y":[],"rating":[],"category":[]}

        for i,row in tqdm(dataset.iterrows(),total=dataset.shape[0]):
            indexed=np.array(self._tokenize(row["text"]))
            label=row["label"]
            rating=row["rating"]
            df["text"].append(indexed)
            df["y"].append(label)
            df["rating"].append(rating)
            df["category"].append(row["category"])

        return pd.DataFrame(df)

    def _tokenize(self,sentence:str):

        if self.args.use_custom_vocab:
            return self.tokenizer.encode(sequence=sentence.lower()).ids

        else:
            return self.tokenizer.encode(text=sentence.lower())

    def _balance_with_label(self,dataset:pd.DataFrame):
        # labels=dataset.label.value_counts().keys())
        subset={}
        for i,row in dataset.iterrows():
            label=row["label"]
            if label not in subset:
                subset[label]=[]
            subset[label].append(row)
        min_value=min([len(example) for key,example in  subset.items()])
        out=[]

        for _, example in subset.items():
            sampled=random.sample(example,min_value)
            out.extend(sampled)

        return pd.DataFrame(out)




class ContrastiveProcessor(object):
    def __init__(self,tokenizer:BertTokenizer,args):
        self.tokenizer=tokenizer
        self.label_map=None
        self.args=args
    def _load_pickle(self,file_path):
        """

        :param file_path:
        :return: pd.DataFrame
        """
        # with open(file_path, 'rb') as f:
        #     df = pickle.load(f)

        # return df
        return pd.read_pickle(file_path)


    def sample(self,df:pd.DataFrame,n):
        print(df.shape)

        if df.shape[0]<n:
            n=df.shape[0]

        return df.sample(n=n).reset_index(drop=True)


    def preprocess(self,file_path:str):
        dataset=self._load_pickle(file_path)

        df={"text":[],"y":[],"augmented":[]}

        for i,row in tqdm(dataset.iterrows(),total=dataset.shape[0]):
            indexed = np.array(self._tokenize(row["original"]))
            augmented = [np.array(self._tokenize(row["aug_{0}".format(i)])) for i in range(0,3)]

            label=row["label"]
            df["text"].append(indexed)
            df["augmented"].append(augmented)
            df["y"].append(label)

        return pd.DataFrame(df)

    def _tokenize(self,sentence:str):

        if self.args.use_custom_vocab:
            return self.tokenizer.encode(sequence=sentence.lower()).ids

        else:
            return self.tokenizer.encode(text=sentence.lower())
