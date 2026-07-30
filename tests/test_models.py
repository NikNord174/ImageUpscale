import pytest
import torch

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
