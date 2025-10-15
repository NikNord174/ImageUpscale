from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np
import torch
from argus import load_model
import os
import matplotlib.pyplot as plt

from src.datatools.utils import resolve_tuple
from src.datatools.read_write_up import ReadWriteUp


CONFIG_PATH = '/workdir/configs/predict_configs.yaml'

OmegaConf.register_new_resolver('tuple', resolve_tuple)


@hydra.main(
        version_base=None, config_path=os.path.dirname(CONFIG_PATH),
        config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def predict(cfg: DictConfig) -> None:
    """Upscale patterns using a trained model."""
    # Upload the dataset to predict
    DEVICE = cfg.model.params.device
    patterns = ReadWriteUp().read_up_file(
        file_path=cfg.predict.data[0],
        dtype=np.uint16)

    patterns = patterns[0].astype(np.float64)
    patterns = torch.from_numpy(patterns).to(DEVICE)

    # Load the model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.predict.model.file_path is None:
        raise ValueError("Model filepath isn't provided in the configuration.")
    model = load_model(
        cfg.predict.model.file_path[0], device=DEVICE,
        optimizer=None)

    patterns = patterns[np.newaxis, ...].to(DEVICE)
    pred = model.predict(patterns.float())

    fig, ax = plt.subplots(1, 2)
    print('Patterns shape: ', patterns.shape)
    print('Pred shape: ', pred.shape)
    ax[0].imshow(patterns[0][0].cpu().numpy(), cmap='gray')
    ax[1].imshow(pred[0][0].cpu().numpy(), cmap='gray')
    plt.savefig(cfg.predict.save_path[0])


if __name__ == '__main__':
    predict()
