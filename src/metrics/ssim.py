import torch
import torch.nn as nn

C1 = 0.01 ** 2
C2 = 0.03 ** 2


class GlobalSSIM(nn.Module):
    """SSIM over whole-image statistics, rescaled to [0, 1].

    This is a simplification of the standard windowed SSIM: mean,
    variance and covariance are computed once per image instead of in
    sliding local windows, which makes it cheap enough to run on every
    batch. Values are therefore not comparable to published SSIM
    numbers; it is a training monitor, not a benchmark metric.
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
