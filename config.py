"""
Author : Chiyeong Heo
Date : 2021-08-17
Reference : https://github.com/hyunwoongko/transformer/blob/master/conf.py
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model hyperparameter
num_layers = 6
d_model = 512
num_heads = 8
d_hidden = 2048
drop_rate = 0.1
batch_size = 128
max_seq_len = 256

# Optimizer hyperparameter (Adam optimizer)
beta1 = 0.9
beta2 = 0.98
eps = 1e-9
warmup_steps = 4000
init_lr = 1e-5
weight_decay = 5e-4
# Learning rate scheduler hyperparameter
lr_factor = 0.9
lr_patience = 10