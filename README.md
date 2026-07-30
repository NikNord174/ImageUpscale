# ImageUpscale

[![ci](https://github.com/NikNord174/ImageUpscale/actions/workflows/ci.yml/badge.svg)](https://github.com/NikNord174/ImageUpscale/actions/workflows/ci.yml)

4x super-resolution of EBSD diffraction patterns with a U-Net.

An EBSD detector in a scanning electron microscope records thousands of
Kikuchi diffraction patterns per scan. Recording them at low resolution
makes the scan much faster, at the cost of pattern detail. This project
trains a U-Net to restore 4x-downsampled patterns back to full
resolution, so a scan could be acquired fast and upscaled afterwards.

![Before and after](docs/results.png)

*Held-out scan, never seen in training. The patterns are private lab
data (polycrystalline nickel), so the repo ships no data or weights;
this figure was produced with the training command below.*

## How it works

- **Data.** Patterns are read straight from `.up2` files, the EDAX
  binary format: a 16-byte header (version, pattern width, pattern
  height, byte offset) followed by back-to-back 16-bit patterns.
  `src/datatools/up_dataset.py` memory-maps the file, so a
  multi-gigabyte scan costs almost no RAM. Patterns are 16-bit and stay
  16-bit: values are scaled to float straight from uint16, never
  squeezed through a uint8 path.
- **Pairs.** Training is self-supervised: each pattern resized to
  128x128 is the target, and the target average-pooled 4x4 is the
  32x32 input. Block averaging rather than smooth resampling, because
  binning is what the detector actually does when it trades resolution
  for speed — and it makes the task honest: a smoothly downsampled
  image keeps so much information that plain bicubic interpolation is
  nearly optimal, while binned patterns leave the network something
  real to recover. No labels needed, every scan is its own dataset.
- **Model.** A U-Net (`src/models/unet.py`) with a twist: the encoder
  halves the resolution four times, but the first two decoder stages
  upsample 4x instead of 2x. The decoder ends up two doublings ahead of
  the encoder, which is where the 4x super-resolution comes from. Skip
  tensors are zero-padded up to the decoder resolution before
  concatenation. The network predicts a residual on top of bicubic
  upsampling of its input — interpolation recovers the smooth structure
  for free, so the model only competes on the detail interpolation
  cannot infer, and never does worse than the baseline it starts from.
- **Training.** MSE loss, AdamW, `pytorch-argus` loop with early
  stopping, best-checkpoint saving, LR reduction on plateau, and CSV
  logging. Model selection monitors SSIM on a *held-out scan* — a
  different scan of the sample, not a slice of the training scan, since
  neighbouring points of one scan sit in the same grain and look nearly
  identical.
- **Metric.** SSIM computed from whole-image statistics
  (`src/metrics/ssim.py`). It is deliberately simpler than the windowed
  SSIM from the literature — cheap enough to run on every batch — and
  the code says so; the numbers are for model selection, not for
  benchmark comparison.

## Results

Trained on one full scan (4,512 patterns, 17 epochs on a laptop GPU),
evaluated on all 3,936 patterns of a second scan of the same sample
covering different grains. The two scans share only the static
detector background: after flat-fielding, validation patterns have no
close match in the training scan (best cross-scan correlation 0.33),
so the score cannot come from memorizing training content.

|                  | PSNR (dB) | global SSIM |
|------------------|-----------|-------------|
| bicubic 4x       | 37.9      | 0.9964      |
| U-Net (residual) | 39.1      | 0.9972      |

The gain is consistent, not statistical: the model reconstructs every
single one of the held-out patterns more accurately than
interpolation. Without the residual connection the same network
*loses* to bicubic on every pattern, which is why the residual design
is there.

## Run it

Needs Python 3.11+. Training runs on CUDA, Apple Silicon (`mps`) or CPU.

```
git clone https://github.com/NikNord174/ImageUpscale
cd ImageUpscale
pip install -r requirements.txt
python -m scripts.train
```

Point the config at your own `.up2` scans first — either edit
`configs/train_configs.yaml` or override on the command line:

```
python -m scripts.train \
    'data.train=["data/scan6.up2"]' \
    'data.valid=["data/scan5.up2"]' \
    model.params.device=mps
```

Checkpoints, logs and the resolved config land in
`data/experiments/<experiment>_<run>/`.

With Docker (GPU host): `make build && make run`.

## Tests

`make test` — the suite runs without any pattern data: it writes a tiny
synthetic `.up2` file and checks the parser, the pair shapes, 16-bit
value preservation, the 4x output geometry of the model, a full
CPU training step, and the SSIM metric. `make lint` runs flake8.

## Limitations

- Both scans are the same material (nickel) on the same instrument;
  cross-material generalization is untested.
- The binning model covers the resolution loss but not the change in
  noise statistics a genuinely faster exposure would bring.
- The SSIM here is a global-statistics variant; don't compare its
  values against published windowed-SSIM numbers.
- The zero-padded skip alignment leaves faint ripple artifacts in the
  network output, visible on close inspection of the figure.
