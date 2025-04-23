import os
import logging

import hydra

from argus import load_model
from omegaconf import OmegaConf, DictConfig
from argus.callbacks import LoggingToCSV, EarlyStopping, LoggingToFile, \
    MonitorCheckpoint

from src.datatools.utils import resolve_tuple
from src.metrics.ssim_metric import SSIM
from src.datatools.get_loader import get_loader


CONFIG_PATH = '/workdir/configs/train_configs.yaml'

OmegaConf.register_new_resolver('tuple', resolve_tuple)


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH),
            config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def train(cfg: DictConfig) -> None:
    logger = logging.getLogger('train')
    experiment_name = cfg.metadata.experiment_name
    run_name = cfg.metadata.run_name
    save_dir = f'/workdir/data/experiments/{experiment_name}_{run_name}'
    model = hydra.utils.instantiate(cfg.model)
    callbacks = [
        EarlyStopping(patience=cfg.train_params.early_stopping_epochs,
                      monitor=cfg.train_params.monitor_metric,
                      better=cfg.train_params.monitor_metric_better),
        MonitorCheckpoint(save_dir, max_saves=1,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better,
                          optimizer_state=True),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
        LoggingToCSV(os.path.join(save_dir, 'stat.csv'))
    ]
    metrics = [SSIM()]

    pretrain_path = cfg.model.params.pretrain
    if pretrain_path is not None:
        if os.path.exists(pretrain_path):
            model = load_model(
                pretrain_path, device=cfg.model.params.device)
            model.set_lr(cfg.model.params.optimizer.lr)
        else:
            logger.error('Could not find pretrain file %s' % pretrain_path)
    train_loader = get_loader(
        data_params=cfg.data.data_params,
        data_paths=cfg.data.train,
        shuffle=True)
    valid_loader = get_loader(
        data_params=cfg.data.data_params,
        data_paths=cfg.data.valid,
        shuffle=False)

    OmegaConf.save(cfg, os.path.join(save_dir, 'train_config.yaml'))
    model.fit(train_loader, val_loader=valid_loader,
              num_epochs=cfg.train_params.max_epochs,
              callbacks=callbacks, metrics=metrics, metrics_on_train=True)


if __name__ == '__main__':
    train()
