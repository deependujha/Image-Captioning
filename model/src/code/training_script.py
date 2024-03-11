import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------- 1. Import the necessary modules -----------------
from caption_vocab import MyVocab
from datasets import get_train_test_dataloader
from vision_transformer_encoder import ViTEncoder
from caption_generator_decoder import ImageCaptionDecoder
from train_and_eval_epoch import train_epoch, evaluate


# ----------------- 2. Define the constants -----------------

# RANDOM_SEED = 42
EPOCHS = 2

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
EMBED_DIM = 256  # embedding dimension
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
    train_df, test_df = train_test_split(df, test_size=0.02, random_state=42)

    train_loader, test_loader = get_train_test_dataloader(
        train_df, test_df, IMAGE_DIR, batch_size=BATCH_SIZE
    )
    return (
        train_loader,
        test_loader,
        train_df,
    )  # train_df is required to get the vocabulary


def _my_vocab(my_df: pd.DataFrame, column_name: str = " comment"):
    """
    Function to get the vocabulary and tokenizer
    """
    my_vocab = MyVocab(df=my_df, column_name=column_name)

    # update the TGT_VOCAB_SIZE
    global TGT_VOCAB_SIZE
    TGT_VOCAB_SIZE = len(my_vocab)  # update the TGT_VOCAB_SIZE
    return my_vocab


if __name__ == "__main__":
    """Steps:
    1. Get the train and test dataloaders ✅
    2. Create and get the vocabulary ✅
    3. Create the Encoder model ✅
    4. Create the Decoder model ✅
    5. Choose the loss function & optimizer for `encoder model parameters` & `decoder model parameters` ✅
    6. Be able to make one training step and one evaluation step ✅
    7. Train the model for X epochs ✅
    """
    train_dataloader, test_dataloader, train_df = _my_train_and_test_dataloader()
    for images, captions in train_dataloader:
        print(images.shape)
        print(captions)
        break
    print("-" * 80)
    my_vocab = _my_vocab(train_df)
    print(f"{len(my_vocab)=}")
    print(my_vocab.PAD_IDX, my_vocab.BOS_IDX, my_vocab.EOS_IDX, my_vocab.UNK_IDX)
    print("-" * 80)
    encoder_model = ViTEncoder(
        num_patches=NUM_PATCHES,
        image_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        in_channels=IN_CHANNELS,
        activation=ENCODER_ACTIVATION,
    ).to(DEVICE)
    random_image = torch.randn(BATCH_SIZE, 3, 224, 224).clip(0, 1).to(DEVICE)
    print(f"{BATCH_SIZE=}")
    print(f"{random_image.shape=}")

    print(encoder_model(random_image).shape)  # BATCH_SIZE X (NUM_PATCHES+1) X EMBED_DIM
    print("-" * 80)
    print(f"{TGT_VOCAB_SIZE=}")
    decoder_model = ImageCaptionDecoder(
        tgt_vocab_size=TGT_VOCAB_SIZE,
        emb_size=EMBED_DIM,
        nhead=NUM_HEADS,
        num_decoder_layers=NUM_ENCODERS,
        dropout=DROPOUT,
        tgt_max_len=TGT_MAX_LEN,
    ).to(DEVICE)
    # final_output = my_img_caption_decoder(
    #     trg=meow_meow,
    #     memory=dummy_encoder_output,
    #     tgt_mask=my_subsequent_mask,
    #     tgt_key_padding_mask=my_padding_mask,
    # )
    print(
        "couldn't run the decoder model as the trg, memory, tgt_mask, tgt_key_padding_mask are not available yet"
    )
    print("-" * 80)
    loss_fn = torch.nn.NLLLoss(ignore_index=my_vocab.PAD_IDX)
    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=LEARNING_RATE)
    print("loss_fn, encoder_optimizer, decoder_optimizer are created")
    print("-" * 80)

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
        print(f"for epoch: {epoch} ==> {average_train_batch_loss=}; {average_val_batch_loss=}")
        print("\n" + "-" * 80 + "\n")
    print("train_epoch is called")
    print("-" * 80)
