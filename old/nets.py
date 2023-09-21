import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets_ import AdaINResBlock, ResBlock
from nets_ import ConvBlock
import torchvision

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = out + skip

        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class Generator(nn.Module):
    def __init__(
        self,
        input_channel=3,
        style_dim=1024,
        res_blocks=6,
        skip=False
    ):
        super().__init__()

        self.style_dim = style_dim

        self.DownResBlks = nn.ModuleList()
        self.MiddleResBlks = nn.ModuleList()
        self.UpResBlks = nn.ModuleList()
        self.ToRGBs = nn.ModuleList()

        self.input_layer = nn.Conv2d(input_channel, 64, kernel_size=1)
        self.skip = skip
        
        # Encoder
        for in_channel, out_channel in [[64,128], [128,256], [256,512], [512,512]]:
            self.DownResBlks.append(ResBlock(in_channel, out_channel, scale_factor=.5))
        
        # Res Blocks
        for _ in range(res_blocks):
            self.MiddleResBlks.append(ResBlock(512, 512, scale_factor=1))
        
        # Decoder
        if skip:
            for in_channel, out_channel in [[1024,512], [1024,256], [512,128], [256,64]]:
                self.UpResBlks.append(AdaINResBlock(in_channel, out_channel, scale_factor=2, style_dim=style_dim))
                self.ToRGBs.append(nn.Conv2d(out_channel, 3, kernel_size=3, stride=1, padding=1))
        else:
            for in_channel, out_channel in [[512,512], [512,256], [256,128], [128,64]]:
                self.UpResBlks.append(AdaINResBlock(in_channel, out_channel, scale_factor=2, style_dim=style_dim))
                self.ToRGBs.append(nn.Conv2d(out_channel, 3, kernel_size=3, stride=1, padding=1))
    
    def forward(self, image, style):
        feat = self.input_layer(image)
        if self.skip:
            feats = []
        
        for DownResBlk in self.DownResBlks:
            feat = DownResBlk(feat)
            if self.skip:
                feats.append(feat)
                
        for MiddleResBlk in self.MiddleResBlks:
            feat = MiddleResBlk(feat)
        
        rgb = torch.zeros_like(feat[:, :3, :, :])
        for UpResBlk, ToRgb in zip(self.UpResBlks, self.ToRGBs):
            if self.skip:
                feat = torch.cat((feat, feats[-1]), dim=1)
            feat = UpResBlk(feat, style)

            rgb_add = ToRgb(feat)

            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear')
            rgb = rgb + rgb_add
            if self.skip:
                feats.pop()

        return rgb

class MyGenerator(nn.Module):
    def __init__(self, n_mlp=8, lr_mlp=0.01, skip=False, ref_input=False):
        super(MyGenerator, self).__init__()
        if ref_input:
            self.synthesis = Generator(style_dim=512, input_channel=6, skip=skip)
        else:
            self.synthesis = Generator(style_dim=512+512, input_channel=3, skip=skip)
        self.skip = skip
        self.ref_input = ref_input
        
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(256, 512, kernel_size=3, stride=3, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(512, 512, kernel_size=1, stride=(3, 1), padding=0),
            Conv2d(512, 512, kernel_size=1, stride=(3, 1), padding=0),
            Conv2d(512, 512, kernel_size=1, stride=(3, 1), padding=0),)
        
        # expression
        self.Expression_Encoder = torchvision.models.mobilenet_v2(pretrained=True)
        layers = [PixelNorm()]
        layers += [EqualLinear(1000, 512, lr_mul=lr_mlp, activation="fused_lrelu")]
        for i in range(n_mlp-1):
            layers.append(EqualLinear(512, 512, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.expression_mapping = nn.Sequential(*layers)

    def get_exp_vector(self, image):
        exp_vector = self.Expression_Encoder(image)
        exp_vector_512 = self.expression_mapping(exp_vector)
        
        return exp_vector_512
    
    def forward(self, input_img, ref_img, audio_feature):
        if self.ref_input:
            cat_img = torch.cat([input_img, ref_img], dim=1)
            # print('input :', input_img.shape) # (BxT, 6, H, W)
            
            audio_embedding = self.audio_encoder(audio_feature) # (B, 1, 768, 9) -> (B, 512, 1, 1)
            audio_embedding = audio_embedding.view(audio_embedding.shape[0], -1) # (B, 512, 1, 1) -> (B, 512)
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1).repeat(5, 1) # (B, 512) -> (B*T, 512)
            
            outputs = self.synthesis(cat_img, audio_embedding)
        
        else:
            audio_embedding = self.audio_encoder(audio_feature) # (B, 1, 768, 9) -> (B, 512, 1, 1)
            audio_embedding = audio_embedding.view(audio_embedding.shape[0], -1) # (B, 512, 1, 1) -> (B, 512)
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1).repeat(5, 1) # (B, 512) -> (B*T, 512)
            
            ref_vector = self.get_exp_vector(ref_img) # (B*T, 512) + (B*T, 512) = (B*T*2, 512)
            cat_vector = torch.cat((audio_embedding, ref_vector), dim = -1)
            outputs = self.synthesis(input_img, cat_vector)
        
        return outputs