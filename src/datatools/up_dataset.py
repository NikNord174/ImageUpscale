import os
import struct

import numpy as np
from torch.utils.data.dataset import Dataset

from src.datatools.transforms import resize_image_torch


class UpDataset(Dataset):
    def __init__(
            self, file_path: str,
            img_size: tuple[int, int] = (128, 128),
            transform=None):
        super().__init__()
        assert os.path.exists(file_path)
        self.train_size = (img_size[0] // 4, img_size[1] // 4)
        self.target_size = img_size
        self.raw_pats = self._read_up_file(file_path)
        self.pats = self.substract_background(self.raw_pats)
        self.transform = transform

    @staticmethod
    def substract_background(data: np.ndarray,) -> np.ndarray:
        """Substracts background from patterns.

        Args:
            data (np.ndarray): Array of patterns with a shape (N, 1, m, m).

        Returns:
            np.ndarray: Patterns with meaned cross and shape (N, 1, m, m).
        """
        dtype = np.uint16
        background = np.mean(data, axis=0).astype(np.float32)

        subtracted_pats = data.astype(np.float32) - background

        original_mean = np.mean(data, axis=(1, 2,), keepdims=True)\
            .astype(np.float32)
        subtracted_mean = np.mean(subtracted_pats, axis=(1, 2,),
                                  keepdims=True)

        adjusted_pats = subtracted_pats + (original_mean - subtracted_mean)
        adjusted_pats = np.clip(adjusted_pats, 0, 65535)

        adjusted_pats = adjusted_pats.astype(dtype)

        return adjusted_pats

    @staticmethod
    def _read_up_file(
            file_path: str,
            dtype: str = np.uint16) -> np.ndarray:
        with open(file_path, 'rb') as up_file:
            header = struct.unpack('4i', up_file.read(16))
            width = header[1]  # Width of patterns in pixels
            height = header[2]  # Height of patterns in pixels
            offset = header[3]  # Offset to first pattern
            pats = np.fromfile(up_file, dtype=dtype, offset=offset-16)
            num_pats = int(pats.shape[0] / (width * height))
        return pats.reshape(num_pats, height, width)

    def __len__(self) -> int:
        return len(self.pats[-1])

    def __getitem__(self, idx: int):
        # Add a new axis for the channel
        image = self.pats[idx]
        target = self.pats[idx]
        image = resize_image_torch(image, self.train_size)
        target = resize_image_torch(target, self.target_size)
        return image, target
