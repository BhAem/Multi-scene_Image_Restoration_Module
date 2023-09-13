import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_model(args, parent=False):
    return Net()


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


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
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


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class LSTM(nn.Module):
    def __init__(self, channel, iteration=2):
        # f_recurrent(LSTM)
        super(LSTM, self).__init__()
        self.iteration = iteration

        self.conv_i = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, row, col = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        h = Variable(torch.zeros(batch_size, channel, row, col))
        c = Variable(torch.zeros(batch_size, channel, row, col))

        x = torch.cat((x, h), 1)
        for i in range(self.iteration):
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

        return h


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

    def fuseforward(self, input):
        for module in self:
            if module is None or type(module) is nn.BatchNorm2d:
                continue
            input = module(input)
        return input



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand: None
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int,
            o: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class Net(nn.Module):
    def __init__(self, res_blocks=2):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.lstm0 = LSTM(16, 2)
        self.dense0 = nn.Sequential(
            # ResidualBlock(16),
            # ResidualBlock(16),
            # ResidualBlock(16)
            InvertedResidual(16, 16, 1, 4),
            InvertedResidual(16, 16, 1, 4)
        )
        self.calayer0 = CALayer(16)
        self.palayer0 = PALayer(16)

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.lstm1 = LSTM(32)
        self.dense1 = nn.Sequential(
            # ResidualBlock(32),
            # ResidualBlock(32),
            # ResidualBlock(32)
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 4)
        )
        self.calayer1 = CALayer(32)
        self.palayer1 = PALayer(32)

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.lstm2 = LSTM(64)
        self.dense2 = nn.Sequential(
            # ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64)
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4)
        )
        self.calayer2 = CALayer(64)
        self.palayer2 = PALayer(64)

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.lstm3 = LSTM(128)
        self.dense3 = nn.Sequential(
            # ResidualBlock(128),
            # ResidualBlock(128),
            # ResidualBlock(128)
            InvertedResidual(128, 128, 1, 4),
            InvertedResidual(128, 128, 1, 4)
        )
        self.calayer3 = CALayer(128)
        self.palayer3 = PALayer(128)

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        # self.fusion4 = Encoder_MDCBlock1(256, 5, mode='iter2')
        #self.dense4 = Dense_Block(256, 256)

        # self.lstm = LSTM(256)
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            # self.dehaze.add_module('res%d' % i, ResidualBlock(256))
            self.dehaze.add_module('res%d' % i, InvertedResidual(256, 256, 1, 4))


        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            # ResidualBlock(128),
            # ResidualBlock(128),
            # ResidualBlock(128)
            InvertedResidual(128, 128, 1, 4),
            InvertedResidual(128, 128, 1, 4)
        )
        self.calayer_4 = CALayer(128)
        self.palayer_4 = PALayer(128)

        # self.fusion_4 = Decoder_MDCBlock1(128, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            # ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64)
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4)
        )
        self.calayer_3 = CALayer(64)
        self.palayer_3 = PALayer(64)
        # self.fusion_3 = Decoder_MDCBlock1(64, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            # ResidualBlock(32),
            # ResidualBlock(32),
            # ResidualBlock(32)
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 4)
        )
        self.calayer_2 = CALayer(32)
        self.palayer_2 = PALayer(32)
        # self.fusion_2 = Decoder_MDCBlock1(32, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            # ResidualBlock(16),
            # ResidualBlock(16),
            # ResidualBlock(16)
            InvertedResidual(16, 16, 1, 4),
            InvertedResidual(16, 16, 1, 4)
        )
        self.calayer_1 = CALayer(16)
        self.palayer_1 = PALayer(16)
        # self.fusion_1 = Decoder_MDCBlock1(16, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x, return_feat=False):
        features = []

        res1x = self.conv_input(x)
        res1x = self.lstm0(res1x)
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(res1x)
        # res2x = self.lstm1(res2x)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        # res4x = self.lstm2(res4x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        # res8x = self.lstm3(res8x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)

        # res16x = self.lstm(res16x)

        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, res1x)
        x = self.dense_1(x) + x - res2x
        x = self.conv_output(x)

        if return_feat:
            return x, features
        else:
            return x


if __name__=='__main__':

    from torchsummary import summary
    net = Net()
    summary(net, (3, 640, 640), device='cpu')

    from thop import profile
    input = torch.randn(1, 3, 640, 640)
    flops, params = profile(net, inputs=(input,))
    print(flops)
    print(params)

