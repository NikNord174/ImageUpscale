import torch
import torch.nn as nn
import torch.nn.functional as F

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    line = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = torch.outer(line, line)
    return kernel / kernel.sum()


class WindowedSSIM(nn.Module):
    """Standard SSIM under a sliding 11x11 Gaussian window (Wang 2004).

    Local means, variances and covariance are measured per window, so
    band edges and fine contrast count towards the score - unlike
    GlobalSSIM, which only sees whole-image statistics. Values are
    comparable to published SSIM numbers for images in [0, 1].
    Expects single-channel input.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.register_buffer(
            'window', gaussian_kernel(window_size, sigma)[None, None])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Full precision on purpose: under fp16 autocast the variance
        # E[x^2] - mu^2 cancels down to the same magnitude as C2.
        x = x.float()
        y = y.float()
        window = self.window.to(device=x.device)
        mu_x = F.conv2d(x, window)
        mu_y = F.conv2d(y, window)
        var_x = F.conv2d(x * x, window) - mu_x ** 2
        var_y = F.conv2d(y * y, window) - mu_y ** 2
        cov = F.conv2d(x * y, window) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)
                    / ((mu_x ** 2 + mu_y ** 2 + C1)
                       * (var_x + var_y + C2)))
        return ssim_map.mean(dim=[1, 2, 3])


class MSESSIMLoss(nn.Module):
    """MSE plus a windowed-SSIM term.

    MSE by itself blurs: the optimum for an uncertain pixel is the
    mean of all candidates. The SSIM term puts a price on losing
    local structure.
    """

    def __init__(self, ssim_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = WindowedSSIM()
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        ssim = self.ssim(prediction, target).mean()
        return (self.mse(prediction, target)
                + self.ssim_weight * (1.0 - ssim))


class GlobalSSIM(nn.Module):
    """SSIM over whole-image statistics, rescaled to [0, 1].

    Simplification of the windowed SSIM: mean, variance and
    covariance are computed once per image instead of in sliding
    local windows, cheap enough to run on every batch. Use it to
    compare epochs of one run; the values are not comparable to
    published SSIM numbers.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dims = [1, 2, 3]
        mu_x = torch.mean(x, dim=dims)
        mu_y = torch.mean(y, dim=dims)
        var_x = torch.var(x, dim=dims, unbiased=False)
        var_y = torch.var(y, dim=dims, unbiased=False)
        cov = torch.mean(x * y, dim=dims) - mu_x * mu_y
        ssim_num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
        ssim_den = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
        return (ssim_num / ssim_den + 1.0) / 2.0
