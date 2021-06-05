import pandas as pd
import random
import numpy as np
import collections
import torch
from torch.utils.data import IterableDataset, DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer



class BatchGenerator(IterableDataset):
    def __init__(self,df:pd.DataFrame,batch_size:int=16,padding_index:int=0,max_length=512):
        self.df=df
        # self.lens=dataset.shape[0]
        self.batch_size=batch_size
        self.padding_index=padding_index
        self.max_length=max_length

    def collate(self,input):
        """
        :param input:
        :return:
        """
        return NotImplementedError


    def __iter__(self):

        return NotImplementedError



from .preprocessor import InputExample,MaskAugmentedExample

class CFBatchGenerator(BatchGenerator):
    def __init__(self,df,batch_size,padding_index,dataset_type="iid",shuffle=False):
        """
        aug_type:: callable class
        """
        super(CFBatchGenerator, self).__init__(df,batch_size,padding_index)

        self.dataset_type = dataset_type
        self.df["lens"]= [len(text) for text in self.df.text]
        self.df = self.sort(self.df)
        num_buckets=len(self.df) // self.batch_size + (len(self.df) % self.batch_size != 0)
        self.num_buckets=num_buckets
        self.dataset_lens=num_buckets
        if shuffle:
            self.df=self.shuffle(self.df,self.num_buckets)


    def sort(self, df, criteria="lens"):
        return df.sort_values(criteria,ascending=False).reset_index(drop=True)

    def _maxlens_in_first_batch(self,df):
        first_batch = df.iloc[0:self.batch_size]

        return first_batch

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(1,num_buckets -1):
            new_df = df.iloc[bucket * self.batch_size: (bucket + 1) * self.batch_size]
            dfs.append(new_df)

        dfs.append(df.iloc[(num_buckets-1) * self.batch_size: num_buckets * self.batch_size])
        random.shuffle(dfs,)

        first_batch=self._maxlens_in_first_batch(df)
        dfs.insert(0,first_batch)

        df = pd.concat(dfs)

        return df.reset_index(drop=True)

    @overrides
    def __iter__(self):
        for _,row in self.df.iterrows():
            example = InputExample(text_a=np.array(row["text"]),label=row["y"])
            yield example.text_a[:self.max_length],example.label


    def __len__(self):
        return self.dataset_lens

    @overrides
    def collate(self,input):
        text_ids=[torch.LongTensor(item[0]) for item in input]
        labels = [torch.LongTensor([item[-1]]) for item in input]
        text_ids=pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        labels=pad_sequence(labels)

        return torch.LongTensor(text_ids),torch.LongTensor(labels)[0]



class BatchGeneratorForTranslation(BatchGenerator):
    def __init__(self,df:pd.DataFrame,batch_size:int=16,padding_index:int=0,max_length=512):
        super(BatchGeneratorForTranslation, self).__init__(df,batch_size,padding_index,max_length)
        self.df["lens"] = [len(text) for text in self.df["input_ids"]]


        self.df = self.sort(self.df)


        num_buckets=len(self.df["input_ids"]) // self.batch_size + (len(self.df) % self.batch_size != 0)
        self.num_buckets=num_buckets
        self.dataset_lens=num_buckets


    def sort(self, df, criteria="lens"):
        return df.sort_values(criteria,ascending=False).reset_index(drop=True)

    def _maxlens_in_first_batch(self,df):
        first_batch = df.iloc[0:self.batch_size]

        return first_batch

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(1,num_buckets -1):
            new_df = df.iloc[bucket * self.batch_size: (bucket + 1) * self.batch_size]
            dfs.append(new_df)

        dfs.append(df.iloc[(num_buckets-1) * self.batch_size: num_buckets * self.batch_size])
        random.shuffle(dfs,)

        first_batch=self._maxlens_in_first_batch(df)
        dfs.insert(0,first_batch)

        df = pd.concat(dfs)

        return df.reset_index(drop=True)

    @overrides
    def __iter__(self):
        for _,row in self.df.iterrows():
            yield row["input_ids"][:self.max_length],row["attention_mask"][:self.max_length],row["label"]


    def __len__(self):
        return self.num_buckets

    @overrides
    def collate(self,input):
        text_ids=[torch.LongTensor(item[0]) for item in input]
        attention_mask = [torch.LongTensor(item[1]) for item in input]
        label = [torch.LongTensor([item[2]]) for item in input]

        text_ids=pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return torch.LongTensor(text_ids),torch.LongTensor(attention_mask),torch.LongTensor(label)









import random


class BatchGeneratorContrastive(CFBatchGenerator):
    def __init__(self, df, batch_size, padding_index, aug_type="None", dataset_type="iid", shuffle=False,**kwargs):
        CFBatchGenerator.__init__(self,df, batch_size, padding_index, dataset_type, shuffle)
        self.aug_type = aug_type

    @overrides
    def __iter__(self):
        for _,row in self.df.iterrows():
            input1,input2= random.sample(row["augmented"],k=2)

            yield input1,input2,row["y"]

    def __len__(self):
        return self.dataset_lens

    @overrides
    def collate(self,input):
        text_id1=[torch.LongTensor(item[0]) for item in input]
        text_id2 = [torch.LongTensor(item[1]) for item in input]

        labels = [torch.LongTensor([item[2]]) for item in input]

        text_id1 = pad_sequence(text_id1, batch_first=True, padding_value=self.padding_index)
        text_id2 = pad_sequence(text_id2, batch_first=True, padding_value=self.padding_index)


        return (torch.LongTensor(text_id1),torch.LongTensor(text_id2)),torch.LongTensor(labels)