import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.resnet18 import ResNet


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        a = torch.max(x, 1)[0].unsqueeze(1)
        b = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((a, b), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):  # 2 80 128 80
        x_compress = self.compress(x)  # 2 2 128 80
        x_out = self.conv(x_compress)  # 2 1 128 80
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        no_spatial=False
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # 2 128 80 80 -> 2 80 128 80
        x_out1 = self.ChannelGateH(x_perm1)  #
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # 2 128 80 80
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # 2 80 80 128
        x_out2 = self.ChannelGateW(x_perm2)  # 2 80 80 128
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # 2 128 80 80
        if not self.no_spatial:
            x_out = self.SpatialGate(x) # 2 128 80 80
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


def upsample(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self, pretrained=False, pretrain_url="", res_blocks=2):
        super(Net, self).__init__()

        self.encoder = None
        self.encoder = ResNet()

        if pretrained:
            print("Loading pretrained weight!!")
            cached_file = pretrain_url
            state_dict = torch.load(cached_file)
            new_state_dict = self.encoder.state_dict()
            # 删除pretrained_dict.items()中model所没有的东西
            state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数
            new_state_dict.update(state_dict)
            self.encoder.load_state_dict(new_state_dict)

        # self.dehaze = nn.Sequential()
        # for i in range(0, res_blocks):
        #     self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.upconv3_0 = ConvBlock(256, 128)
        self.upconv3_1 = ConvBlock(256, 128)
        # self.dense_3 = nn.Sequential(
        #     # ResidualBlock(16),
        #     ResidualBlock(128),
        #     ResidualBlock(128)
        # )
        self.dispconv3 = Conv3x3(128, 3)

        self.upconv2_0 = ConvBlock(128, 64)
        self.upconv2_1 = ConvBlock(128, 64)
        # self.dense_2 = nn.Sequential(
        #     # ResidualBlock(16),
        #     ResidualBlock(64),
        #     ResidualBlock(64)
        # )
        self.dispconv2 = Conv3x3(64, 3)

        self.upconv1_0 = ConvBlock(64, 32)
        self.upconv1_1 = ConvBlock(96, 32)
        # self.dense_1 = nn.Sequential(
        #     # ResidualBlock(16),
        #     ResidualBlock(32),
        #     ResidualBlock(32)
        # )
        self.dispconv1 = Conv3x3(32, 3)

        self.upconv0_0 = ConvBlock(32, 16)
        self.upconv0_1 = ConvBlock(16, 16)
        self.dispconv0 = Conv3x3(16, 3)

        self.triplet_attention3 = TripletAttention()
        self.triplet_attention2 = TripletAttention()
        self.triplet_attention1 = TripletAttention()


    def forward(self, input_image, return_feat=False):
        outputs = []
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))

        x = self.features[-1]
        x = self.upconv3_0(x)
        x = [upsample(x)]
        x += [self.features[3 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv3_1(x)
        x = self.triplet_attention3(x)
        # x = self.dense_3(x)
        outputs.append(upsample(self.dispconv3(x), 8))

        x = self.upconv2_0(x)
        x = [upsample(x)]
        x += [self.features[2 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)
        x = self.triplet_attention2(x)
        # x = self.dense_2(x)
        outputs.append(upsample(self.dispconv2(x), 4))

        x = self.upconv1_0(x)
        x = [upsample(x)]
        x += [self.features[1 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)
        x = self.triplet_attention1(x)
        # x = self.dense_1(x)
        outputs.append(upsample(self.dispconv1(x), 2))

        x = self.upconv0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv0_1(x)
        x = self.dispconv0(x)
        outputs.append(x)

        if return_feat:
            return x, outputs
        else:
            return x


if __name__=='__main__':
    model = Net(True, r"../weights/resnet18-f37072fd.pth", 2)
    # image = torch.randn(4, 3, 192, 640)
    # output = model(image)
    # print(output)

    # from torchsummary import summary
    from torchinfo import summary
    summary(model, (2, 3, 640, 640), device='cpu')

    from thop import profile
    input = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)