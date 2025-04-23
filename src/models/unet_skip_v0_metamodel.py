import torch
import torch.nn as nn
from argus import Model
from argus.utils import deep_to, deep_detach

from src.models.unet_skip_v0 import UNet


class UNetMetaModel(Model):
    nn_module = UNet
    optimizer = torch.optim.AdamW
    loss = nn.MSELoss
    device = 'cpu'

    def __init__(self, params: dict):
        super().__init__(params)
        self.amp = bool(params.get('amp', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()

        x, target = deep_to(batch, self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
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
            'loss': loss.item()
        }

    def val_step(self, batch, state) -> dict:
        self.eval()
        with torch.no_grad():
            x, target = deep_to(batch, device=self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(x)
            loss = self.loss(prediction, target)
        prediction = deep_detach(prediction)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': deep_detach(target),
            'loss': loss.item()
        }
