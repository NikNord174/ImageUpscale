"""Train the upscaling model.

Reads configs/train_configs.yaml; any value can be overridden on the
command line, e.g.:

    python -m scripts.train train_params.max_epochs=5 \
        model.params.device=cpu
"""

import logging
import os

import hydra
from argus import load_model
from argus.callbacks import (
    EarlyStopping,
    LoggingToCSV,
    LoggingToFile,
    MonitorCheckpoint,
    ReduceLROnPlateau,
)
from omegaconf import DictConfig, OmegaConf

from src.datatools.get_loader import get_loader
from src.datatools.utils import resolve_tuple
from src.metrics.ssim_metric import SSIM

OmegaConf.register_new_resolver('tuple', resolve_tuple)


@hydra.main(version_base=None, config_path='../configs',
            config_name='train_configs')
def train(cfg: DictConfig) -> None:
    logger = logging.getLogger('train')
    save_dir = os.path.join(
        cfg.train_params.save_root,
        f'{cfg.metadata.experiment_name}_{cfg.metadata.run_name}')

    pretrain_path = cfg.model.params.pretrain
    if pretrain_path:
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(pretrain_path)
        model = load_model(pretrain_path, device=cfg.model.params.device)
        model.set_lr(cfg.model.params.optimizer.lr)
        logger.info('Loaded pretrain %s', pretrain_path)
    else:
        # _convert_ makes hydra pass plain containers, not DictConfig:
        # argus stores the params in every checkpoint, and torch.load
        # refuses non-primitive types under its weights_only default.
        model = hydra.utils.instantiate(cfg.model, _convert_='all')

    monitor = cfg.train_params.monitor_metric
    better = cfg.train_params.monitor_metric_better
    callbacks = [
        EarlyStopping(
            patience=cfg.train_params.early_stopping_epochs,
            monitor=monitor, better=better),
        MonitorCheckpoint(
            save_dir, max_saves=1, monitor=monitor, better=better,
            optimizer_state=True),
        ReduceLROnPlateau(
            monitor=monitor, better=better,
            factor=cfg.train_params.reduce_lr_factor,
            patience=cfg.train_params.reduce_lr_patience),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
        LoggingToCSV(os.path.join(save_dir, 'stat.csv')),
    ]

    train_loader = get_loader(
        data_params=cfg.data.data_params,
        data_paths=cfg.data.train,
        shuffle=True)
    valid_loader = get_loader(
        data_params=cfg.data.data_params,
        data_paths=cfg.data.valid)

    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, 'train_config.yaml'))
    model.fit(
        train_loader, val_loader=valid_loader,
        num_epochs=cfg.train_params.max_epochs,
        callbacks=callbacks, metrics=[SSIM()], metrics_on_train=True)


if __name__ == '__main__':
    train()
