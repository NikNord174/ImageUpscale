from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np
import torch, torchvision
from argus import load_model
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

from src.datatools.utils import resolve_tuple
from src.models.unet_skip_v0_metamodel import UNetMetaModel
from src.datatools.up_dataset import UpDataset
from src.datatools.read_write_up import ReadWriteUp
from src.datatools.transforms import resize_image_torch, to_tensor


CONFIG_PATH = '/workdir/configs/predict_configs.yaml'

OmegaConf.register_new_resolver('tuple', resolve_tuple)

# pats = UpDataset(
#     file_path='/workdir/data/SP_poly-Ni_scan6_15kV_100pA_WD8-1_bin1_sat0100_exp140ms_gain00_gut00_step3500nm_2303points_.up2',
#     img_size=(128, 128),
#     transform=None
# )._read_up_file('/workdir/data/SP_poly-Ni_scan6_15kV_100pA_WD8-1_bin1_sat0100_exp140ms_gain00_gut00_step3500nm_2303points_.up2')

# model = argus.model.load_model(
#     model=UNetMetaModel,
#     file_path='/workdir/data/experiments/test target resize_6/model-094-0.722198.pth',
#     device='cuda:0'
# )

# output = model.predict(pats[0:1])
# print('Output shape: ', output.shape)



@hydra.main(
        version_base=None, config_path=os.path.dirname(CONFIG_PATH),
        config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def predict(cfg: DictConfig) -> None:
    """Upscale patterns using a trained model."""
    # Upload the dataset to predict
    DEVICE = cfg.model.params.device
    # patterns = UpDataset(
    #     file_path=cfg.predict.data[0],
    #     img_size=cfg.data.data_params.img_size,
    #     transform=None)_rea
    patterns = ReadWriteUp().read_up_file(
        file_path=cfg.predict.data[0],
        dtype=np.uint16)

    # resize = torchvision.transforms.Resize((128, 128))
    patterns = patterns[0].astype(np.float64)
    len = patterns.shape[0]
    # patterns = [cv2.resize(pattern, (128, 128)) for pattern in patterns[:10]]
    # patterns = torchvision.transforms.functional.to_tensor(patterns[0].reshape((1, 129, 129)))
    patterns = torch.from_numpy(patterns).to(DEVICE)
    print('Pattens type: ', patterns.dtype)
    print('Patterns shape: ', patterns.shape)

    # Load the model
    model = hydra.utils.instantiate(cfg.model)
    model = load_model(
        cfg.predict.model.file_path[0], device=DEVICE)
    
    patterns = patterns[np.newaxis, ...].to(DEVICE)
    print('Patterns shape: ', patterns.shape)
    pred = model.predict(patterns.float())
    print('Pred shape: ', pred.shape)

    # predictions = []
    # batch_size = 8
    # # patterns = torch.randn(10, 1, 32, 32)
    # for i in tqdm(range(0, len(patterns), 1)):
    #     # pattern = patterns[i][np.newaxis, ...].to(DEVICE)
    #     batch = patterns[i:i+batch_size].to(DEVICE)
    #     pred = model.predict(batch)
    #     predictions.append(pred)
    # results = torch.stack(predictions)
    # print('Results shape: ', results.shape)
    fig, ax = plt.subplots(1, 2)
    print('Patterns shape: ', patterns.shape)
    print('Pred shape: ', pred.shape)
    ax[0].imshow(patterns[0][0].cpu().numpy(), cmap='gray')
    ax[1].imshow(pred[0][0].cpu().numpy(), cmap='gray')
    plt.savefig(cfg.predict.save_path[0])

    # save the predicted results to a file
    # filename = cfg.predict.save_path[0]
    # print('Results shape: ', results.shape)
    # ReadWriteUp().write_up_file(
    #     pat_size=cfg.data.data_params.img_size[0],
    #     file_path=filename,
    #     write_pats=results,
    #     dtype=np.uint16
    # )
    # try:
    #     with open(filename, "w") as file:
    #         for row in results:
    #             file.write(" ".join(map(str, row[0].tolist())) + "\n")
    #     print(f"Results written to {filename} successfully.")
    # except Exception as e:
    #     print(f"Error writing to file: {e}")
    # return results


if __name__ == '__main__':
    predict()
