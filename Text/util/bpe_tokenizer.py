import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from tokenizers import BertWordPieceTokenizer
logger=logging.getLogger(__name__)


class CustomTokenizer:
    def __init__(self,dir_path,vocab_size,encoder_class):
        self.dir_path=dir_path
        self.encoder=BertWordPieceTokenizer()
        self.vocab_size=vocab_size
        self.encoder=self.load_encoder(encoder_class,dir_path,vocab_size)

        self.mask_token_id=self.encoder.token_to_id("[MASK]")
    @classmethod
    def _pkl_to_txt(cls):

        data_dir=os.path.join(cls.dir_path)
        logger.info("Flatten Corpus to text file")
        df = pd.read_pickle(os.path.join(data_dir,"train.pkl"))

        txt_file=os.path.join(data_dir,"train.txt")
        f=open(txt_file,"w")
        textlines=[row["text"].lower().replace("\n"," ") for i,row in df.iterrows()]

        for textline in tqdm(textlines):
            f.write(textline+"\n")

        f.close()


    @classmethod
    def train(cls):
        cls._pkl_to_txt()
        txt_path=os.path.join(cls.dir_path,"train.txt")

        cls.encoder.train(txt_path, vocab_size=cls.vocab_size)
        cls.encoder.save(cls.vocab_dir)


    @classmethod
    def load_encoder(cls, dir_path, vocab_size):
        encoder_class=BertWordPieceTokenizer
        cls.encoder=BertWordPieceTokenizer()
        cls.vocab_size=vocab_size
        cls.dir_path=dir_path
        cls.vocab_dir=os.path.join(cls.dir_path,"custom-vocab")

        if not os.path.isdir(cls.vocab_dir):
            os.makedirs(cls.vocab_dir)

        vocab_name = os.path.join(cls.vocab_dir ,'vocab.txt')
        if os.path.exists(vocab_name):
            logger.info('\ntrained encoder loaded')
            # self.istrained = True
            encoder = encoder_class(vocab_file=vocab_name)
            encoder.mask_token_id = encoder.token_to_id("[MASK]")

            return encoder
        else:
            # self.istrained = False
            logger.info('\nencoder needs to be trained')
            cls.train()
            cls.encoder.mask_token_id = cls.encoder.token_to_id("[MASK]")
            return cls.encoder
