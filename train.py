"""
Author : Chiyeong Heo
Date : 2021-08-19
"""

from torch import nn, optim
from torch.optim import Adam

from config import *
from model.whole_model.transformer import Transformer
from util.dataloader import DataLoader

import torchtext

loader = DataLoader(SRC_LANGUAGE='en',
                    TGT_LANGUAGE='de')

src_lang = 'en'
tgt_lang = 'de'

train_data, valide_data, test_data = loader.load_dataset()
loader.build_vocab(dataset=train_data, min_freq=2)

model = Transformer(pad = loader.PAD_IDX,
                    enc_voc_size = len(loader.vocabs[src_lang]),
                    dec_voc_size = len(loader.vocabs[tgt_lang]),
                    d_model = d_model, 
                    max_seq_len = max_seq_len,
                    drop_rate = drop_rate,
                    n_heads = num_heads,
                    d_hidden = d_hidden,
                    n_layers = num_layers,
                    device = device)

optimizer = Adam(params = model.parameters(),
                 lr = init_lr,
                 betas = (beta1, beta2),
                 eps = eps,
                 weight_decay = weight_decay)