import numpy as np
import torch
import torchvision.transforms as transforms

UINT16_MAX = 65535.0
UINT8_MAX = 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert one pattern to a (1, H, W) float tensor in [0, 1].

    .up2 patterns are 16-bit, so the scale has to follow the dtype:
    routing them through a uint8 path would wrap the values modulo 256
    and destroy the upper byte of every pixel.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'expected np.ndarray, got {type(image).__name__}')
    scale = UINT16_MAX if image.dtype == np.uint16 else UINT8_MAX
    array = image.astype(np.float32) / scale
    return torch.from_numpy(array).unsqueeze(0)


def resize_image_torch(
        image: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize one pattern to target_size, returned as a float tensor."""
    resize = transforms.Resize(target_size, antialias=True)
    return resize(to_tensor(image))
