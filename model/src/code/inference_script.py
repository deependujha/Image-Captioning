import os
import torch
import numpy as np

from caption_vocab import MyVocab
from vision_transformer_encoder import VitEncoder
from caption_generator_decoder import ImageCaptionDecoder


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


def inference_encoder_decoder_model(
    encoder_model, decoder_model, image, my_vocab, max_target_length=20, DEVICE="cpu"
):
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(DEVICE)
        encoder_output = encoder_model(image)

        my_subsequent_mask = my_vocab.create_square_subsequent_mask(
            max_target_length
        ).to(
            DEVICE
        )  # max_len

        my_caption = [my_vocab.BOS_IDX]
        for curr_idx in range(max_target_length):
            padded_tgt = (
                torch.tensor(
                    my_caption
                    + ([my_vocab.PAD_IDX] * (max_target_length - len(my_caption)))
                )
                .unsqueeze(0)
                .to(DEVICE)
            )
            my_padding_mask = my_vocab.create_padding_mask(padded_tgt).to(DEVICE)

            final_output = decoder_model(
                trg=padded_tgt,
                memory=encoder_output,
                tgt_mask=my_subsequent_mask,
                tgt_key_padding_mask=my_padding_mask,
            )

            final_output = final_output.argmax(2)

            # update tgt
            my_caption = my_caption + [final_output[0][curr_idx].item()]
            # sentence = my_vocab.get_sentence_from_indices(tgt)
            # print(f"new {sentence=}")
            if my_caption[-1] == my_vocab.EOS_IDX:
                break

        sentence = my_vocab.get_sentence_from_indices(my_caption)
        return sentence


# inference functions for AWS SageMaker endpoint
# - model_fn
# - input_fn
# - predict_fn
# - output_fn


def model_fn(model_dir):
    print(f"loading model for inference from {model_dir=}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        model = torch.load(f, map_location=DEVICE)
        encoder_model.load_state_dict(model["encoder"], device=DEVICE)
        decoder_model.load_state_dict(model["decoder"], device=DEVICE)
        vocab = model["vocab"]

    return encoder_model, decoder_model, vocab


def predict_fn(input_data, model, context):
    print(f"predicting on input data {input_data.shape=}")
    device = torch.device(
        "cuda:" + str(context.system_properties.get("gpu_id"))
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder_model = model[0].to(device)
    decoder_model = model[1].to(device)
    my_vocab = model[2]
    inference_encoder_decoder_model(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        image=input_data,
        my_vocab=my_vocab,
    )


def output_fn(prediction, context):
    print("all done, returning output")
    return prediction
