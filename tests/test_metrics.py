import pytest
import torch

from src.metrics.ssim import GlobalSSIM
from src.metrics.ssim_metric import SSIM


def test_identical_images_score_one():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 32, 32)
    ssim = GlobalSSIM()(x, x)
    assert torch.allclose(ssim, torch.ones(4), atol=1e-4)


def test_noise_lowers_the_score():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 32, 32)
    noisy = (x + 0.5 * torch.randn_like(x)).clamp(0, 1)
    ssim = GlobalSSIM()
    assert ssim(x, noisy).mean() < ssim(x, x).mean()


def test_metric_averages_over_samples():
    torch.manual_seed(0)
    metric = SSIM()
    x = torch.rand(3, 1, 16, 16)
    metric.update({'prediction': x, 'target': x})
    assert metric.compute() == pytest.approx(1.0, abs=1e-4)


def test_metric_is_zero_before_any_update():
    metric = SSIM()
    assert metric.compute() == 0.0
    metric.update({'prediction': torch.rand(2, 1, 16, 16),
                   'target': torch.rand(2, 1, 16, 16)})
    metric.reset()
    assert metric.compute() == 0.0
