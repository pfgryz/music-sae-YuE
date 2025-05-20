import random
import re

import numpy as np
import torch
from transformers import LogitsProcessor


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


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def split_lyrics(lyrics: str) -> list[str]:
    """
    Splits the lyrics

    Returns
    -------
    list[str]
        List of lyrics
    """
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics
