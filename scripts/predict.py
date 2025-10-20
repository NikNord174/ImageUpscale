from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np
import torch
from argus import load_model
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from src.datatools.utils import resolve_tuple
from src.datatools.transforms import resize_image_torch
from src.datatools.read_write_up import ReadWriteUp
from src.metrics.ssim import SSIM_v0 as ssim


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

    patterns = patterns[0][0]
    im_size = cfg.data.data_params.img_size
    patterns_down = resize_image_torch(patterns, im_size)
    patterns_orig = resize_image_torch(patterns, tuple(x*4 for x in im_size))

    # Load the model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.predict.model.file_path is None:
        raise ValueError("Model filepath isn't provided in the configuration.")
    model = load_model(
        cfg.predict.model.file_path[0], device=DEVICE,
        optimizer=None)

    patterns = patterns_down[np.newaxis, ...].to(DEVICE)
    pred = model.predict(patterns.float())

    fig, ax = plt.subplots(1, 2)
    ssim_value = ssim()(patterns_orig[np.newaxis, ...], pred)
    print('ssim value for prediction: ', ssim_value)
    ax[0].imshow(patterns[0][0].cpu().numpy(), cmap='gray')
    ax[1].imshow(pred[0][0].cpu().numpy(), cmap='gray')
    plt.savefig(cfg.predict.save_path[0])


if __name__ == '__main__':
    predict()
