import torch

from src.models.unet_skip_v0 import UNet


# Testing the model
if __name__ == "__main__":
    # Create a UNet model
    model = UNet(n_channels=1, n_classes=1)
    print('Model: ', model)

    # Create a test input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 1, 128, 128)
    print(f"Input shape: {x.shape}")

    # Pass the input through the model
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")