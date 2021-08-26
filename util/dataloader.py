"""
Author : Chiyeong Heo
Date : 2021-08-19
Reference : https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List


"""
Dependency
    > pip install -U spacy
    > python -m spacy download en_core_web_sm
    > python -m spacy download de_core_news_sm
"""

class TranslationDataloader:
    def __init__(self, SRC_LANGUAGE, TGT_LANGUAGE):
        # SRC_LANGUAGE and TGT_LANGUAGE must be 'de' or 'en'
        self.src_lan = SRC_LANGUAGE
        self.tgt_lan = TGT_LANGUAGE
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3

        self.tokenizers = {}
        self.vocabs = {}

    def load_dataset(self):
        train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'),
                                                     language_pair=(self.src_lan, self.tgt_lan))

        return train_data, valid_data, test_data

    def yield_tokens(self, data_iter, language):
        lan_idx = {self.src_lan: 0, self.tgt_lan: 1}

        for data in data_iter:
            yield self.tokenizers[language](data[lan_idx[language]])

    def build_vocab(self, dataset, min_freq):
        # Tokenize
        if self.src_lan == 'de' and self.tgt_lan == 'en':
            self.tokenizers[self.src_lan] = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
            self.tokenizers[self.tgt_lan] = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        else:
            src_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
            tgt_tokenizer = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
        
        self.vocabs[self.src_lan] = build_vocab_from_iterator(iterator=self.yield_tokens(dataset, self.src_lan),
                                                              min_freq = min_freq,
                                                              specials = self.special_symbols, 
                                                              special_first=True)

        self.vocabs[self.tgt_lan] = build_vocab_from_iterator(iterator=self.yield_tokens(dataset, self.tgt_lan),
                                                              min_freq=min_freq,
                                                              specials=self.special_symbols,
                                                              special_first=True)
        # For handling OOV problem
        self.vocabs[self.src_lan].set_default_index(self.UNK_IDX)
        self.vocabs[self.tgt_lan].set_default_index(self.UNK_IDX)

        
   # def make_batched_iter(self, dataset, batch_size):
    #    def collate_batch(batch):
     #       src_batch, tgt_batch = [], []
      #      for src_sample, tgt_sample in batch:
       #         src_batch.append()



    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
    
    # tokens -> BOS + tokens + EOS
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))

    

    # token_transform : tokenizers
    # vocab_transform : vocabs
    # text_transform  : sequence를 tokenizer로 tokenize 후 vocab으로 만든 다음 BOS/EOS 더해 tensor화
    def make_batched_iter(self, batch):
        text_transform = {}
        text_transform[self.src_lan] = self.sequential_transforms(self.tokenizers[self.src_lan],
                                                                  self.vocabs[self.src_lan],
                                                                  self.tensor_transform)
        def collate_batch(batch):
            src_batch, tgt_batch = [], []
            for src_sample, tgt_sample in batch:
                src_batch.append(text_transform[self.src_lan](src_sample.rstrip("\n")))
                tgt_batch.append(text_transform[self.tgt_lan](tgt_sample.rstrip("\n")))

            src_batch = pad_sequence(sequences=src_batch, padding_value=self.PAD_IDX)
            tgt_batch = pad_sequence(sequences=tgt_batch, padding_value=self.PAD_IDX)
            return src_batch, tgt_batch

        batched_iter = DataLoader()