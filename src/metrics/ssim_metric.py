import torch
from argus.metrics import Metric

from src.metrics.ssim import GlobalSSIM


class SSIM(Metric):
    name = 'ssim'
    better = 'max'

    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.value = 0.0
        self.metric = GlobalSSIM()

    def reset(self):
        self.n_samples = 0
        self.value = 0.0

    def update(self, step_output: dict):
        prediction = step_output['prediction']
        target = step_output['target']
        ssim = self.metric(target, prediction)
        self.n_samples += prediction.shape[0]
        self.value += torch.sum(ssim).item()

    def compute(self):
        if self.n_samples == 0:
            return 0.0
        return self.value / self.n_samples
