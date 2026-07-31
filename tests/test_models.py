import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.metamodel import UNetMetaModel
from src.models.unet import UNet


@pytest.mark.parametrize('size', [16, 32])
def test_output_is_4x_input(size):
    model = UNet(n_channels=1, o_channels=1)
    model.eval()
    x = torch.randn(2, 1, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, size * 4, size * 4)


def test_decoder_parameters_are_registered():
    model = UNet()
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith('conv1') for name in names)
    assert any(name.startswith('conv4') for name in names)
    assert any(name.startswith('outc') for name in names)


def test_zeroed_head_returns_plain_bicubic():
    model = UNet()
    nn.init.zeros_(model.outc.weight)
    nn.init.zeros_(model.outc.bias)
    model.eval()
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        out = model(x)
    expected = F.interpolate(
        x, scale_factor=4, mode='bicubic', align_corners=False)
    # With the head silenced only the residual wiring remains.
    assert torch.allclose(out, expected, atol=1e-6)


def test_metamodel_train_step_runs_on_cpu():
    torch.manual_seed(0)
    model = UNetMetaModel({
        'nn_module': {'n_channels': 1, 'o_channels': 1},
        'optimizer': {'lr': 1e-3},
        'device': 'cpu',
    })
    batch = (torch.randn(2, 1, 16, 16), torch.randn(2, 1, 64, 64))
    out = model.train_step(batch, None)
    assert out['prediction'].shape == (2, 1, 64, 64)
    assert isinstance(out['loss'], float)


def test_metamodel_val_step_detaches():
    torch.manual_seed(0)
    model = UNetMetaModel({
        'nn_module': {'n_channels': 1, 'o_channels': 1},
        'optimizer': {'lr': 1e-3},
        'device': 'cpu',
    })
    batch = (torch.randn(2, 1, 16, 16), torch.randn(2, 1, 64, 64))
    out = model.val_step(batch, None)
    assert out['prediction'].shape == (2, 1, 64, 64)
    assert not out['prediction'].requires_grad


def test_config_instantiates_on_cpu():
    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    import scripts.train  # noqa: F401  registers the tuple resolver

    cfg = OmegaConf.load(
        Path(__file__).parents[1] / 'configs' / 'train_configs.yaml')
    cfg.model.params.device = 'cpu'
    model = hydra.utils.instantiate(cfg.model, _convert_='all')
    assert type(model).__name__ == 'UNetMetaModel'
    data = OmegaConf.to_container(cfg.data, resolve=True)
    assert tuple(data['data_params']['img_size']) == (128, 128)
