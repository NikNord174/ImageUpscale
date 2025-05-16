import torch

from src.models.unet_skip_v0 import UNet
from src.models.unet_skip_v1 import UNet as UNetV1


# Testing the model
if __name__ == "__main__":
    # Create a UNet model
    device = 'cpu'
    # model = UNet(n_channels=[1], o_channels=[1]).to(device)
    modelV1 = UNetV1(n_channels=[1], o_channels=[1]).to(device)
    # print('Model: ', modelV1)

    # Create a test input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 1, 32, 32).to(device)
    # print(f"Input shape: {x.shape}")
    # print('Input device: ', x.get_device())
    # print('Model device: ', next(model.parameters()).get_device())
    # print('ModelV1 device: ', next(modelV1.parameters()).get_device())

    # Pass the input through the model
    output = modelV1(x)
    print(f"Output shape: {output.shape}")
    print('----')
    # output = model(x)
    # print(f"Output shape old: {output.shape}")

    # # Print model statistics
    # total_params = sum(p.numel() for p in modelV1.parameters())
    # print(f"\nTotal parameters: {total_params:,}")

    # trainable_params = sum(p.numel()
    #                        for p in modelV1.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {trainable_params:,}")
