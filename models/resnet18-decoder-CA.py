import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.resnet18 import ResNet


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()  # 2 128 80 80
        x_h = self.pool_h(x)  # 2 128 80 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 2 128 1 80 -> 2 128 80 1

        y = torch.cat([x_h, x_w], dim=2)  # 2 128 160 1
        y = self.conv1(y)  # 2 8 160 1
        y = self.bn1(y)  # 2 8 160 1
        y = self.act(y)  # 2 8 160 1

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 2 8 80 1
        x_w = x_w.permute(0, 1, 3, 2)  # 2 8 1 80

        a_h = self.conv_h(x_h).sigmoid()  # 2 128 80 1
        a_w = self.conv_w(x_w).sigmoid()  # 2 128 1 80

        out = identity * a_w * a_h  # 2 128 80 80

        return out


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

        self.ca3 = CoordAtt(128, 128)
        self.ca2 = CoordAtt(64, 64)
        self.ca1 = CoordAtt(32, 32)


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
        x = self.ca3(x)
        # x = self.dense_3(x)
        outputs.append(upsample(self.dispconv3(x), 8))

        x = self.upconv2_0(x)
        x = [upsample(x)]
        x += [self.features[2 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)
        x = self.ca2(x)
        # x = self.dense_2(x)
        outputs.append(upsample(self.dispconv2(x), 4))

        x = self.upconv1_0(x)
        x = [upsample(x)]
        x += [self.features[1 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)
        x = self.ca1(x)
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