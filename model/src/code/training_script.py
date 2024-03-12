"""_summary_
Main training script for image-captioning model
"""

import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------- 1. Import the necessary modules -----------------
from caption_vocab import MyVocab
from utils import count_parameters
from datasets import get_train_test_dataloader
from vision_transformer_encoder import VitEncoder
from caption_generator_decoder import ImageCaptionDecoder
from train_and_eval_epoch import train_epoch, evaluate
from inference_script import inference_encoder_decoder_model


# ----------------- 2. Define the constants -----------------

# RANDOM_SEED = 42
EPOCHS = 20

LEARNING_RATE = 1e-3
PATCH_SIZE = 16
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 8
DROPOUT = 0.01
ENCODER_ACTIVATION = "gelu"
DECODER_ACTIVATION = "relu"
NUM_ENCODERS = 6
NUM_DECODERS = 6
EMBED_DIM = 768  # embedding dimension
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
TGT_MAX_LEN = 20  # maximum length of the target sequence
TGT_VOCAB_SIZE = 0  # to be updated later


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS_DIR = "../../datasets"
CAPTION_CSV_FILE = f"{DATASETS_DIR}/results.csv"
IMAGE_DIR = f"{DATASETS_DIR}/flickr30k_images"
BATCH_SIZE = 4


# ----------------- 3. Main Code -----------------


def _my_train_and_test_dataloader():
    """
    Function to get the train and test dataloaders
    """
    df = pd.read_csv(CAPTION_CSV_FILE, delimiter="|")
    _train_df, _test_df = train_test_split(df, test_size=0.02, random_state=42)

    train_loader, test_loader = get_train_test_dataloader(
        _train_df, _test_df, IMAGE_DIR, batch_size=BATCH_SIZE
    )
    return (
        train_loader,
        test_loader,
        _train_df,
    )  # train_df is required to get the vocabulary


def _my_vocab(my_df: pd.DataFrame, column_name: str = " comment"):
    """
    Function to get the vocabulary and tokenizer
    """
    _my_vocab = MyVocab(df=my_df, column_name=column_name)

    # update the TGT_VOCAB_SIZE
    new_tgt_vocab_size = len(_my_vocab)  # update the TGT_VOCAB_SIZE
    return _my_vocab, new_tgt_vocab_size


def print_number_of_parameters(_encoder_model, _decoder_model):
    """_summary_
    Prints the number of parameters in the encoder and decoder models.
    """
    _encoder_model.eval()
    _decoder_model.eval()

    enc_count = count_parameters(_encoder_model)
    dec_count = count_parameters(_decoder_model)

    print("-" * 80)
    print(f"Number of parameters in the encoder model: {enc_count}")
    print(f"Number of parameters in the decoder model: {dec_count}")
    print(f"Total number of parameters: {enc_count+dec_count}")
    print("-" * 80)


if __name__ == "__main__":

    train_dataloader, test_dataloader, train_df = _my_train_and_test_dataloader()

    my_vocab, TGT_VOCAB_SIZE = _my_vocab(train_df)

    encoder_model = VitEncoder(
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        in_channels=IN_CHANNELS,
        activation=ENCODER_ACTIVATION,
    ).to(DEVICE)

    decoder_model = ImageCaptionDecoder(
        tgt_vocab_size=TGT_VOCAB_SIZE,
        emb_size=EMBED_DIM,
        nhead=NUM_HEADS,
        num_decoder_layers=NUM_ENCODERS,
        dropout=DROPOUT,
        tgt_max_len=TGT_MAX_LEN,
    ).to(DEVICE)

    loss_fn = torch.nn.NLLLoss(ignore_index=my_vocab.PAD_IDX)
    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        average_train_batch_loss = train_epoch(
            epoch_num=epoch,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            loss_fn=loss_fn,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            train_dataloader=train_dataloader,
            DEVICE=DEVICE,
            my_vocab=my_vocab,
            max_target_length=TGT_MAX_LEN,
        )

        average_val_batch_loss = evaluate(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            loss_fn=loss_fn,
            val_dataloader=test_dataloader,
            DEVICE=DEVICE,
            my_vocab=my_vocab,
            max_target_length=TGT_MAX_LEN,
        )
        print(
            f"for epoch: {epoch} ==> {average_train_batch_loss=}; {average_val_batch_loss=}"
        )
        print_number_of_parameters(encoder_model, decoder_model)
        print("\n" + "-" * 80 + "\n")
    print("-" * 80)

    inference_sentence = inference_encoder_decoder_model(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        image=torch.randn(3, 224, 224),
        my_vocab=my_vocab,
        max_target_length=TGT_MAX_LEN,
        DEVICE=DEVICE,
    )

    print(f"{inference_sentence=}")
    print_number_of_parameters(encoder_model, decoder_model)
    print("Done!")
