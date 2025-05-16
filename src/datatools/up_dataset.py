import os
import struct

import numpy as np
from torch.utils.data.dataset import Dataset

from src.datatools.transforms import resize_image_torch, to_tensor


class UpDataset(Dataset):
    def __init__(
            self, file_path: str,
            img_size: tuple[int, int] = (128, 128),
            transform=None):
        super().__init__()
        assert os.path.exists(file_path)
        self.train_size = (img_size[0] // 4, img_size[1] // 4)
        self.target_size = img_size
        self.pats = self._read_up_file(file_path)
        self.transform = transform

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
        # target = to_tensor(self.pats[idx])
        image = self.pats[idx]
        target = self.pats[idx]
        image = resize_image_torch(image, self.train_size)
        target = resize_image_torch(target, self.target_size)
        # print('Image size: ', image.size(), 'Target size: ', target.size())
        return image, target
