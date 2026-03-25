import torch

from loguru import logger
from functools import lru_cache


@lru_cache()
def get_available_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logger.debug(f"Device is set to: \n{device}")
    return device
