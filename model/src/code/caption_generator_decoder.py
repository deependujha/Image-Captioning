import math
import torch
import torch.nn as nn

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
        if emb_size % 2 != 0:
            raise Exception("Embedding size must be even")
        super(PositionalEncoding, self).__init__()

        # fancy logarithmic stuff to finally calculate 10000^(-2i/emb_size)
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))

        pos_embedding[:, 0::2] = torch.sin(
            pos * den
        )  # from 0th index to last index with a step of 2
        pos_embedding[:, 1::2] = torch.cos(
            pos * den
        )  # from 1st index to last index with a step of 2
        # pos_embedding = pos_embedding.unsqueeze(0) # add a new dimension at the first index, we will use batch_first = True (handle batch dimension)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "pos_embedding", pos_embedding
        )  # register the tensor as buffer - not updated during backprop

    def forward(self, token_embedding):
        # no learnable parameters
        return self.dropout(token_embedding + self.pos_embedding)


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class ImageCaptionDecoder(nn.Module):
    """_summary_
        Decoder module for Image Captioning transformer
    """

    def __init__(
        self,
        tgt_vocab_size: int,
        emb_size: int,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        activation: str = "relu",
        dropout: float = 0.1,
        tgt_max_len: int = 20,
    ):
        super(ImageCaptionDecoder, self).__init__()

        image_caption_decoder = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            activation=activation,
            dropout=dropout,
            batch_first=True
        )
        self.image_caption_decoder_block = nn.TransformerDecoder(
            image_caption_decoder, num_layers=num_decoder_layers
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=tgt_max_len)

    def forward(
        self,
        trg,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.image_caption_decoder_block(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        return self.generator(outs)
