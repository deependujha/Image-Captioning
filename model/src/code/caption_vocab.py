from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List, Union
import pandas as pd
import torch
import torch.nn as nn


class MyVocab:
    """_summary_
    A class to create a vocabulary from the captions in the pandas dataframe
    """

    def __init__(self, df: pd.DataFrame, column_name: str = " comment"):
        # Define special symbols and indices
        # UNK - unknown token, when the token is not in the vocabulary
        # PAD - padding of token, when the batch is shorter than the max length
        # BOS - beginning of sentence token
        # EOS - end of sentence token
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.tokenizer = get_tokenizer("basic_english")
        self.my_vocab = self._create_vocab(df, column_name)

    def __len__(self):
        return len(self.my_vocab)

    def _create_vocab(self, df: pd.DataFrame, column_name: str = " comment"):

        # helper function to yield list of tokens
        def yield_tokens(list_of_captions: Iterable):

            for _caption in list_of_captions:
                yield self.tokenizer(_caption)

        # Create torchtext's Vocab object
        vocab_transform = build_vocab_from_iterator(
            yield_tokens(df[column_name].tolist()),
            min_freq=1,
            specials=self.special_symbols,
            special_first=True,
        )

        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        vocab_transform.set_default_index(self.UNK_IDX)
        return vocab_transform

    def get_token_index_from_sentence(
        self, sentences: Union[List[str] | str], max_len: int = 30
    ) -> List[int]:
        """_summary_
        Get the indices of the tokens in the sentence

        Args:
            sentence (str): Your input sentence
            max_len (int, optional): Max length. Defaults to 30.

        Returns:
            List[int]: List of indices
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        tokens = [self.tokenizer(sentence) for sentence in sentences]

        token_indices = []
        for token in tokens:
            indices = (
                [self.BOS_IDX] + [self.my_vocab[tok] for tok in token] + [self.EOS_IDX]
            )
            indices += [self.PAD_IDX] * (max_len - len(indices))
            token_indices.append(indices)
        return torch.tensor(token_indices)

    def get_sentence_from_indices(self, indices: torch.tensor) -> str:
        """_summary_
        Get the sentence from the indices

        Args:
            indices (List[int]): List of indices

        Returns:
            str: Sentence
        """
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(0)
        indices = indices.tolist()
        all_sentences = []
        for index in indices:
            tokens = [self.my_vocab.get_itos()[token] for token in index]
            tokens = " ".join(tokens)
            for special_token in self.special_symbols:
                tokens = tokens.replace(special_token, "")

            all_sentences.append(tokens.strip())

        return all_sentences

    def create_padding_mask(self, tgt):
        tgt_padding_mask = (
            (tgt == self.PAD_IDX).type(torch.FloatTensor).clone().detach()
        )
        return tgt_padding_mask

    def create_square_subsequent_mask(self, sz):
        return nn.Transformer.generate_square_subsequent_mask(sz)
