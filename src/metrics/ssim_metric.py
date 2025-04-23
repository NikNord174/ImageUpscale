import torch
from argus.metrics import Metric

from src.metrics.ssim import SSIM_v0


class SSIM(Metric):
    name = 'ssim'
    better = 'max'

    def __init__(self):
        super().__init__()
        self.n_samples: int = 0
        self.value: float = 0.0
        self.metric = SSIM_v0()

    def reset(self):
        self.n_samples: int = 0
        self.value: float = 0.0

    def update(self, step_output: dict):
        print(self.metric)
        pred = step_output['prediction']
        target = step_output['target']
        mse = self.metric(target, pred)
        self.n_samples += pred.shape[0]
        self.value += torch.sum(mse)

    def compute(self):
        if self.n_samples > 0:
            return self.value / self.n_samples
        else:
            return float('inf')
