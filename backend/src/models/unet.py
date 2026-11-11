"""3D U-Net factory (MONAI). Same architecture for centralized and federated runs."""
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_model(in_channels=4, out_channels=3, channels=(8, 16, 32, 64, 128)):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
    )