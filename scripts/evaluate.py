"""Score the model against bicubic upsampling on a held-out scan.

    python -m scripts.evaluate data/scan_valid.up2 \
        --checkpoint data/experiments/unet_baseline/<model>.pth \
        --device cpu --figure docs/results.png

Prints PSNR and windowed SSIM for bicubic and for the model over
every pattern of the scan; --figure renders the comparison strip
shown in the readme.
"""

import argparse

import torch
import torch.nn.functional as F
from argus import load_model

from src.datatools.up_dataset import SCALE, UpDataset
from src.metrics.ssim import WindowedSSIM
from src.models.metamodel import UNetMetaModel  # noqa: F401  argus scope


def upsample(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        images, scale_factor=SCALE, mode='bicubic',
        align_corners=False).clamp(0, 1)


def evaluate(model, dataset, device, batch_size=64):
    ssim = WindowedSSIM()
    sums = {'bicubic': [0.0, 0.0], 'model': [0.0, 0.0]}
    count = 0
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        pairs = [dataset[i] for i in range(start, stop)]
        x = torch.stack([p[0] for p in pairs])
        target = torch.stack([p[1] for p in pairs])
        with torch.no_grad():
            pred = model.predict(x.to(device)).cpu().clamp(0, 1)
        for name, images in (('bicubic', upsample(x)), ('model', pred)):
            mse = ((images - target) ** 2).mean(dim=[1, 2, 3])
            sums[name][0] += (-10 * torch.log10(mse)).sum().item()
            sums[name][1] += ssim(target, images).sum().item()
        count += stop - start
    return {name: (total[0] / count, total[1] / count)
            for name, total in sums.items()}, count


def render_figure(model, dataset, device, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Show the two patterns with the strongest band structure, judged
    # by variance after subtracting the scan-average background.
    sample = np.stack([dataset[i][1][0].numpy()
                       for i in range(0, len(dataset), 8)])
    flat = sample - sample.mean(axis=0)
    order = np.argsort(flat.reshape(len(flat), -1).var(axis=1))[::-1]
    picks = [int(order[0]) * 8, int(order[len(order) // 20]) * 8]

    def psnr(a, b):
        return -10 * np.log10(float(((a - b) ** 2).mean()))

    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.6))
    titles = ['input', 'bicubic 4x', 'U-Net 4x', 'target']
    for row, index in enumerate(picks):
        image, target = dataset[index]
        with torch.no_grad():
            pred = model.predict(
                image[None].to(device)).cpu().clamp(0, 1)[0]
        bicubic = upsample(image[None])[0]
        low, high = np.percentile(target.numpy(), [1, 99])
        labels = ['', f'PSNR {psnr(bicubic, target):.1f} dB',
                  f'PSNR {psnr(pred, target):.1f} dB', '']
        panels = [image[0], bicubic[0], pred[0], target[0]]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel.numpy(), cmap='gray', vmin=low, vmax=high)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            if labels[col]:
                ax.set_xlabel(labels[col], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('up_file', help='held-out .up2 scan')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--figure', default=None,
                        help='also render the comparison strip here')
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device,
                       optimizer=None, loss=None)
    dataset = UpDataset(file_path=args.up_file)
    results, count = evaluate(model, dataset, args.device)
    print(f'patterns: {count}')
    print(f'{"":10}{"PSNR dB":>10}{"windowed SSIM":>16}')
    for name, (psnr, ssim) in results.items():
        print(f'{name:10}{psnr:10.2f}{ssim:16.3f}')
    if args.figure:
        render_figure(model, dataset, args.device, args.figure)
        print(f'figure written to {args.figure}')


if __name__ == '__main__':
    main()
