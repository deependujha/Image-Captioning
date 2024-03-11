import torch

def inference_encoder_decoder_model(encoder_model, decoder_model, image, my_vocab, max_target_length, DEVICE):
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(DEVICE)
        encoder_output = encoder_model(image)
        # print(f"{encoder_output.shape=}")
        
        
        tgt = torch.tensor([my_vocab.SOS_IDX]).unsqueeze(0).to(DEVICE)
        for _ in range(max_target_length):
            my_padding_mask = my_vocab.create_padding_mask(tgt)
            my_subsequent_mask = my_vocab.create_square_subsequent_mask(
                tgt.shape[1]
            )  # max_len
            final_output = decoder_model(
                trg=tgt,
                memory=encoder_output,
                tgt_mask=my_subsequent_mask,
                tgt_key_padding_mask=my_padding_mask,
            )
            # print(f"{final_output.shape=}")
            final_output = final_output.argmax(2)
            # print(f"{final_output.shape=}")
            # print(f"{final_output=}")
            tgt = torch.cat((tgt, final_output[:, -1].unsqueeze(0)), dim=1)
            # print(f"{tgt.shape=}")
            if final_output[:, -1].item() == my_vocab.EOS_IDX:
                break
        return tgt