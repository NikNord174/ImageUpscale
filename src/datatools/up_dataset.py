import os
import struct

import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from src.datatools.transforms import resize_image_torch

SCALE = 4  # detector binning factor


class UpDataset(Dataset):
    """(low-res, high-res) training pairs from one .up2 pattern file.

    Every pattern in the scan yields a pair: the pattern resized to
    img_size is the target, and the target average-pooled 4x4 is the
    input. The averaging matches the detector, which bins pixel
    blocks when it trades resolution for speed.
    """

    def __init__(
            self, file_path: str,
            img_size: tuple[int, int] = (128, 128)):
        super().__init__()
        assert os.path.exists(file_path), file_path
        self.target_size = tuple(int(s) for s in img_size)
        self.pats = self._read_up_file(file_path)

    @staticmethod
    def _read_up_file(file_path: str, dtype=np.uint16) -> np.ndarray:
        """Memory-map the patterns of an EDAX .up2 file.

        The header is four little-endian int32: version, pattern width,
        pattern height, byte offset of the first pattern. Patterns are
        stored back to back as 16-bit unsigned ints. Mapping instead of
        reading keeps a multi-gigabyte scan out of RAM; __getitem__
        touches one pattern at a time.
        """
        with open(file_path, 'rb') as up_file:
            header = struct.unpack('4i', up_file.read(16))
        width = header[1]
        height = header[2]
        offset = header[3]
        pats = np.memmap(file_path, dtype=dtype, mode='r', offset=offset)
        num_pats = pats.shape[0] // (width * height)
        return pats[:num_pats * width * height].reshape(
            num_pats, height, width)

    def __len__(self) -> int:
        return self.pats.shape[0]

    def __getitem__(self, idx: int):
        pattern = np.asarray(self.pats[idx])
        target = resize_image_torch(pattern, self.target_size)
        image = F.avg_pool2d(target, SCALE)
        return image, target
