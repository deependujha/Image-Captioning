"""
Contains the function to train and evaluate the model for one epoch.
"""


def train_epoch(
    epoch_num,
    encoder_model,
    decoder_model,
    loss_fn,
    encoder_optimizer,
    decoder_optimizer,
    train_dataloader,
    DEVICE,
    my_vocab,
    max_target_length=20,
):
    print(f"Epoch {epoch_num}")

    encoder_model.train()
    decoder_model.train()
    losses = 0

    data_seen_so_far = 0  # will be reset to 0 after every 1000 data points seen
    last_thousand_losses = 0

    for src, tgt in train_dataloader:

        data_seen_so_far += src.shape[0]

        src = src.to(DEVICE)
        tgt = my_vocab.get_token_index_from_sentence(tgt, max_len=max_target_length).to(
            DEVICE
        )
        my_padding_mask = my_vocab.create_padding_mask(tgt).to(DEVICE)
        my_subsequent_mask = my_vocab.create_square_subsequent_mask(
            max_target_length
        ).to(
            DEVICE
        )  # max_len

        encoder_output = encoder_model(src)
        # print(f"{encoder_output.shape=}")

        final_output = decoder_model(
            trg=tgt,
            memory=encoder_output,
            tgt_mask=my_subsequent_mask,
            tgt_key_padding_mask=my_padding_mask,
        )
        # print(f"{final_output.shape=}")

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = loss_fn(final_output.permute(0, 2, 1), tgt)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        losses += loss.item()
        last_thousand_losses += loss.item()

        if data_seen_so_far >= 1000:
            data_seen_so_far = 0
            avg_last_thousand_losses = last_thousand_losses / 1000
            last_thousand_losses = 0
            print(f"    {avg_last_thousand_losses=}")

    return losses / len(train_dataloader)  # average loss per batch


def evaluate(
    encoder_model,
    decoder_model,
    loss_fn,
    val_dataloader,
    my_vocab,
    DEVICE,
    max_target_length=20,
):
    encoder_model.eval()
    decoder_model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = my_vocab.get_token_index_from_sentence(tgt, max_len=max_target_length).to(
            DEVICE
        )
        my_padding_mask = my_vocab.create_padding_mask(tgt).to(DEVICE)
        my_subsequent_mask = my_vocab.create_square_subsequent_mask(
            max_target_length
        ).to(
            DEVICE
        )  # max_len

        encoder_output = encoder_model(src)
        # print(f"{encoder_output.shape=}")

        final_output = decoder_model(
            trg=tgt,
            memory=encoder_output,
            tgt_mask=my_subsequent_mask,
            tgt_key_padding_mask=my_padding_mask,
        )

        loss = loss_fn(final_output.permute(0, 2, 1), tgt)
        losses += loss.item()

    return losses / len(list(val_dataloader))
