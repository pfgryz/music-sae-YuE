import random

import numpy as np
import torch


def initialize_seed(seed: int = 42) -> None:
    """
    Initializes the random seed for reproductibility across common libraries.

    Sets the seed for `random`, NumPy and PyTorch.

    Parameters
    ----------
    seed : int = 42
        The seed value to sued

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
