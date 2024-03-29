import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet18 import ResNet


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


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


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

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.upconv3_0 = ConvBlock(256, 128)
        self.upconv3_1 = ConvBlock(256, 128)
        self.dense_3 = nn.Sequential(
            # ResidualBlock(16),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.dispconv3 = Conv3x3(128, 3)

        self.upconv2_0 = ConvBlock(128, 64)
        self.upconv2_1 = ConvBlock(128, 64)
        self.dense_2 = nn.Sequential(
            # ResidualBlock(16),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.dispconv2 = Conv3x3(64, 3)

        self.upconv1_0 = ConvBlock(64, 32)
        self.upconv1_1 = ConvBlock(96, 32)
        self.dense_1 = nn.Sequential(
            # ResidualBlock(16),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.dispconv1 = Conv3x3(32, 3)

        self.upconv0_0 = ConvBlock(32, 16)
        self.upconv0_1 = ConvBlock(16, 16)
        self.dispconv0 = Conv3x3(16, 3)

        self.calayer_3 = simam_module(128)
        self.palayer_3 = simam_module(128)
        self.calayer_2 = simam_module(64)
        self.palayer_2 = simam_module(64)
        self.calayer_1 = simam_module(32)
        self.palayer_1 = simam_module(32)


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

        # res_dehaze = self.features[-1]
        # in_ft = self.features[-1]*2
        # self.features[-1] = self.dehaze(in_ft) + in_ft - res_dehaze

        x = self.features[-1]
        x = self.upconv3_0(x)
        x = [upsample(x)]
        x += [self.features[3 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv3_1(x)
        # x = self.calayer_3(x)
        x = self.palayer_3(x)
        # x = self.dense_3(x)
        # outputs.append(self.dispconv3(x), 8)

        x = self.upconv2_0(x)
        x = [upsample(x)]
        x += [self.features[2 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)
        # x = self.calayer_2(x)
        x = self.palayer_2(x)
        # x = self.dense_2(x)
        # outputs.append(self.dispconv2(x))

        x = self.upconv1_0(x)
        x = [upsample(x)]
        x += [self.features[1 - 1]]
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)
        # x = self.calayer_1(x)
        x = self.palayer_1(x)
        # x = self.dense_1(x)
        # outputs.append(upsample(self.dispconv1(x), 2))

        x = self.upconv0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv0_1(x)
        x = self.dispconv0(x)
        # outputs.append(x)

        if return_feat:
            return x, outputs
        else:
            return x


if __name__=='__main__':
    model = Net(True, r"../weights/resnet18-f37072fd.pth", 2)
    # image = torch.randn(4, 3, 192, 640)
    # output = model(image)
    # print(output)

    from torchsummary import summary
    summary(model, (3, 640, 640), device='cpu')

    from thop import profile
    input = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)