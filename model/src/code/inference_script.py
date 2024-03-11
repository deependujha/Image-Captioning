import torch

def inference_encoder_decoder_model(encoder_model, decoder_model, image, my_vocab, max_target_length, DEVICE):
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(DEVICE)
        encoder_output = encoder_model(image)

        my_subsequent_mask = my_vocab.create_square_subsequent_mask(
            max_target_length
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
            my_padding_mask = my_vocab.create_padding_mask(padded_tgt)

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
