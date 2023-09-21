import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# ------------------------------------------------------------------------------------------
# ConvBlock
#   1. Upsample / Conv(padding)
#       - padding options : 'zeros'(default), 'reflect', 'replicate' or 'circular'
#       - if you choose upsample option, you have to set stride==1
#   2. Norm
#       - Norm options : 'bn', 'in', 'none'
#   3. activation
#       - activation options : 'relu', 'tanh', 'sig', 'none'
# ------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        norm_type="bn",
        act_type="lrelu",
        transpose=False,
    ):
        super(ConvBlock, self).__init__()

        # convolutional layer and upsampling
        self.up = transpose
        self.scale_factor = stride

        if transpose:
            self.conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            )

        # normalization
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm_type == "in":
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm_type == "none":
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm_type}"

        # activation
        if act_type == "relu":
            self.act = nn.ReLU()
        elif act_type == "lrelu":
            self.act = nn.LeakyReLU(0.2)
        elif act_type == "tanh":
            self.act = nn.Tanh()
        elif act_type == "sig":
            self.act = nn.Sigmoid()
        elif act_type == "none":
            self.act = None
        else:
            assert 0, f"Unsupported activation: {act_type}"

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear")

        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.act:
            x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, norm="bn", act="lrelu"):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_c)
        self.activ1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_c)
        self.activ2 = nn.LeakyReLU(0.2)

        self.conv1x1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="reflect",
            bias=False,
        )

        self.scale_factor = scale_factor
        self.resize = scale_factor != 1

    def forward(self, feat):
        feat1 = feat
        feat1 = self.conv1(feat1)
        feat1 = self.norm1(feat1)
        feat1 = self.activ1(feat1)

        if self.resize:
            feat1 = F.interpolate(
                feat1, scale_factor=self.scale_factor, mode="bilinear"
            )

        feat1 = self.conv2(feat1)
        feat1 = self.norm2(feat1)
        feat1 = self.activ2(feat1)

        # skip connction
        feat2 = feat
        if self.resize:
            feat2 = F.interpolate(
                feat2, scale_factor=self.scale_factor, mode="bilinear"
            )  # size
            feat2 = self.conv1x1(feat2)  # chnnel dim

        return feat1 + feat2


class AdaINResBlock(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, style_dim=512):
        super(AdaINResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.AdaIN1 = AdaIN(style_dim, out_c)
        self.activ1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.AdaIN2 = AdaIN(style_dim, out_c)
        self.activ2 = nn.LeakyReLU(0.2)

        self.conv1x1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="reflect",
            bias=False,
        )

        self.scale_factor = scale_factor
        self.resize = scale_factor != 1

    def forward(self, feat, style):
        feat1 = feat
        feat1 = self.conv1(feat1)
        feat1 = self.AdaIN1(feat1, style)
        feat1 = self.activ1(feat1)

        if self.resize:
            feat1 = F.interpolate(
                feat1, scale_factor=self.scale_factor, mode="bilinear"
            )

        feat1 = self.conv2(feat1)
        feat1 = self.AdaIN2(feat1, style)
        feat1 = self.activ2(feat1)

        # skip connction
        feat2 = feat
        if self.resize:
            feat2 = F.interpolate(
                feat2, scale_factor=self.scale_factor, mode="bilinear"
            )  # size
            feat2 = self.conv1x1(feat2)  # chnnel dim

        return feat1 + feat2
