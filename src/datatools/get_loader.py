from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset

from .up_dataset import UpDataset


def get_loader(
        data_params: DictConfig, data_paths: list[str],
        shuffle: bool = False) -> DataLoader:
    dataset = ConcatDataset([
        UpDataset(file_path=data_path,
                  img_size=data_params.img_size,)
        for data_path in data_paths
    ])
    loader = DataLoader(
        dataset, batch_size=data_params.batch_size,
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        shuffle=shuffle)
    return loader


if __name__ == '__main__':
    get_loader()
