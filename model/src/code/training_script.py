"""_summary_
Main training script for image-captioning model
"""

import argparse
import json
import logging
import os
import sys

# SageMaker data parallel: Import the library PyTorch API
import smdistributed.dataparallel.torch.torch_smddp

# SageMaker data parallel: Import PyTorch's distributed API
import torch.distributed as dist

# SageMaker data parallel: Initialize the process group
dist.init_process_group(backend="smddp")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ------------------------------------------------------------------

import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------- 1. Import the necessary modules -----------------
from caption_vocab import MyVocab
from utils import count_parameters, save_model
from datasets import get_train_test_dataset, get_train_test_dataloader
from vision_transformer_encoder import VitEncoder
from caption_generator_decoder import ImageCaptionDecoder
from train_and_eval_epoch import train_epoch, evaluate
from inference_script import inference_encoder_decoder_model


# ----------------- 2. Define the constants -----------------

# RANDOM_SEED = 42

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


print("-" * 80)
# SM DDP only supports GPU-only training
DEVICE = torch.device("cuda")
print("-" * 80)

# ----------------- 3. Main Code -----------------


def _my_train_and_test_dataloader(_datasets_dir, batch_size):
    """
    Function to get the train and test dataloaders
    """
    caption_csv_file = f"{_datasets_dir}/results.csv"
    image_dir = f"{_datasets_dir}/flickr30k_images"

    df = pd.read_csv(caption_csv_file, delimiter="|")
    _train_df, _test_df = train_test_split(df, test_size=0.02, random_state=42)

    train_dataset, test_dataset = get_train_test_dataset(_train_df, _test_df, image_dir)

    # train_sampler
    # SageMaker data parallel: Set num_replicas and rank in DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )

    train_loader, test_loader = get_train_test_dataloader(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        train_sampler=train_sampler,
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


def orchestrator(args):
    my_world_size = dist.get_world_size()
    logger.info(
        f"Initialized the distributed environment: '{args.backend}' backend on {my_world_size} nodes."
    )

    # SageMaker data parallel: Scale batch size by world size
    batch_size = args.batch_size // dist.get_world_size()
    batch_size = max(batch_size, 1)

    train_dataloader, test_dataloader, train_df = _my_train_and_test_dataloader(
        args.data_dir, batch_size
    )

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

    # SageMaker data parallel: Wrap the PyTorch model with the library's DDP
    encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model)
    decoder_model = torch.nn.parallel.DistributedDataParallel(decoder_model)

    # SageMaker data parallel: Pin each GPU to a single library process.
    local_rank = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(local_rank))
    encoder_model.cuda(int(local_rank))
    decoder_model.cuda(int(local_rank))

    loss_fn = torch.nn.NLLLoss(ignore_index=my_vocab.PAD_IDX)
    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
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
        print("\n" + "-" * 80 + "\n")
    print("-" * 80)
    print_number_of_parameters(encoder_model, decoder_model)
    print("-" * 80)
    logger.info("Saving trained model only on rank 0")
    # SageMaker data parallel: Save model on master node.
    if dist.get_rank() == 0:
        save_model(
            model_dir=args.model_dir,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            vocab=my_vocab,
            logger=logger,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # PyTorch environments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="smddp",
        help="backend for dist. training, this script only supports gloo",
    )

    # SageMaker environment
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    orchestrator(parser.parse_args())
