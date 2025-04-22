import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def to_tensor(image):
    """
    Convert a numpy array to a PyTorch tensor

    Args:
        image: input image
    """
    if isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
    transform = transforms.ToTensor()
    return transform(pil_img)


def resize_image_torch(image, target_size):
    """
    Resize an image using PyTorch transforms

    Args:
        image: input image
        target_size: Tuple of (height, width) for the target size
    """
    image = image.astype(np.uint8)
    pil_img = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(pil_img)
