"""
Author : Chiyeong Heo
Date : 2021-08-11
"""

from torch import nn

from model.embedding.PositionalEncoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    """
    Token Embedding + Positional Encoding
    """
    def __init__(self, vocab_size, d_model, max_seq_len, dropout_rate, device):
        super(TransformerEmbedding, self).__init__()

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=1, device = device)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, device=device)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        token_emb = self.token_emb(x)
        pos_enc = self.pos_enc(x)
        return self.dropout(token_emb + pos_enc)