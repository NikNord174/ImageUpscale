import struct

import numpy as np
import pytest
import torch

from src.datatools.transforms import to_tensor
from src.datatools.up_dataset import UpDataset

WIDTH = 64
HEIGHT = 64
NUM_PATS = 6


@pytest.fixture()
def up_file(tmp_path):
    """A small synthetic .up2 file: 4-int32 header + uint16 patterns."""
    rng = np.random.default_rng(0)
    pats = rng.integers(
        0, 65535, size=(NUM_PATS, HEIGHT, WIDTH), dtype=np.uint16)
    pats[0, :, :] = 40000  # constant marker above the uint8 range
    path = tmp_path / 'scan.up2'
    with path.open('wb') as f:
        f.write(struct.pack('4i', 1, WIDTH, HEIGHT, 16))
        f.write(pats.tobytes())
    return path, pats


def test_len_counts_patterns(up_file):
    path, _ = up_file
    dataset = UpDataset(file_path=str(path), img_size=(64, 64))
    assert len(dataset) == NUM_PATS


def test_pair_shapes_and_range(up_file):
    path, _ = up_file
    dataset = UpDataset(file_path=str(path), img_size=(64, 64))
    image, target = dataset[1]
    assert image.shape == (1, 16, 16)
    assert target.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert 0.0 <= target.min() and target.max() <= 1.0


def test_parser_preserves_16_bit_values(up_file):
    path, pats = up_file
    dataset = UpDataset(file_path=str(path), img_size=(64, 64))
    assert np.array_equal(np.asarray(dataset.pats), pats)
    # A uint8 conversion would wrap 40000 to 40000 % 256 = 64.
    _, target = dataset[0]
    assert target.max() == pytest.approx(40000 / 65535, abs=1e-3)


def test_to_tensor_rejects_non_arrays():
    with pytest.raises(TypeError):
        to_tensor([[1, 2], [3, 4]])
