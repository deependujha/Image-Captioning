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


# Patch embedding module - splits the image into patches and then embeds them.
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,  # take each patch_size pixel
            ),
            nn.Flatten(2),  # flatten from the 2nd dimension to the end
        )

        # special classification token
        self.special_classification_token = nn.Parameter(
            torch.randn(size=(1, 1, embed_dim)), requires_grad=True
        )

        self.position_embeddings = PositionalEncoding(
            emb_size=embed_dim,
            dropout=dropout,
            maxlen=num_patches + 1,  # add the in_channels to the num_patches
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.special_classification_token.expand(
            x.shape[0], -1, -1
        )  # (B, input_channel, E)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings(x)
        x = self.dropout(x)
        return x


# Vision Transformer Encoder module
class ViTEncoder(nn.Module):
    def __init__(
        self,
        num_patches,
        image_size,
        patch_size,
        embed_dim,
        num_encoders,
        num_heads,
        dropout,
        in_channels,
        activation="gelu",
    ):
        super().__init__()
        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.encoder_blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoders
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        # x is of shape (B, num_patches+1, embed_dim)
        # x is now the contextually aware embedding representation of the image
        return x
