"""_summary_
    Utility functions like save_model, get_number_of_parameters_in_model
"""

import os
import torch


def save_model(model, model_dir, is_distributed, logger):
    """_summary_
    Saves the model to the specified directory.
    """
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")

    if is_distributed:
        # model now refers to the DistributedDataParallel (DDP) object
        # which is a wrapper around the model you passed to it
        # to get the original model, you can use `model.module`
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def count_parameters(model):
    """Calculates the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: The total number of parameters in the model.
    """
    model.eval()
    return sum(p.numel() for p in model.parameters())
