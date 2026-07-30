import torch
from argus import Model
from argus.utils import deep_detach, deep_to

from src.metrics.ssim import MSESSIMLoss
from src.models.unet import UNet


class UNetMetaModel(Model):
    nn_module = UNet
    optimizer = torch.optim.AdamW
    loss = MSESSIMLoss
    device = 'cuda'

    def __init__(self, params: dict):
        super().__init__(params)
        self.amp = bool(params.get('amp', False))
        self.device_type = torch.device(str(self.device)).type
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.amp)

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()
        x, target = deep_to(batch, self.device, non_blocking=True)
        with torch.amp.autocast(self.device_type, enabled=self.amp):
            prediction = self.nn_module(x)
            loss = self.loss(prediction, target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        prediction = deep_detach(prediction)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': deep_detach(target),
            'loss': loss.item(),
        }

    def val_step(self, batch, state) -> dict:
        self.eval()
        with torch.no_grad():
            x, target = deep_to(batch, self.device, non_blocking=True)
            with torch.amp.autocast(self.device_type, enabled=self.amp):
                prediction = self.nn_module(x)
                loss = self.loss(prediction, target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item(),
        }
