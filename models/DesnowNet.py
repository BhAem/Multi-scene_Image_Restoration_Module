import torch
import torch.nn as nn


class Pyramid_maxout(nn.Module):
    def __init__(self, in_channel=385, depth=3, beta=4):
        super(Pyramid_maxout, self).__init__()
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(in_channel, depth, 2 * i + 1, 1, padding=i))
        self.activation = nn.PReLU(num_parameters=depth)
        self.conv_module = nn.ModuleList(block)

    def forward(self, f):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                conv_result = module(f).unsqueeze(0)
            else:
                temp = module(f).unsqueeze(0)
                conv_result = torch.cat([conv_result, temp], dim=0)
        result, _ = torch.max(conv_result, dim=0)
        return self.activation(result)


class R_t(nn.Module):
    # The recovery submodule (Rt) of the translucency recovery (TR) module
    def __init__(self, in_channel=385, beta=4):
        super(R_t, self).__init__()
        self.SE = Pyramid_maxout(in_channel, 1, beta)
        self.AE = Pyramid_maxout(in_channel, 3, beta)


    def forward(self, x, f_t, **kwargs):
        z_hat = self.SE(f_t)
        a_hat = self.AE(f_t)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            z = kwargs['mask']
            a = kwargs['a']
        elif 'mask' in kwargs.keys():
            z = kwargs['mask']
            a = a_hat
        else:
            z = z_hat
            a = a_hat
        # yield estimated snow-free image y'
        y_ = (z < 1) * (x - a_hat * z) / (1 - z + 1e-8) + (z == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            with torch.no_grad():
                y = (z < 1) * (x - a * z) / (1 - z + 1e-8) + (z == 1) * x
            f_c = torch.cat([y, z, a], dim=1)
        elif 'mask' in kwargs.keys():
            f_c = torch.cat([y_, z, a], dim=1)
        else:
            f_c = torch.cat([y_, z, a], dim=1)
        return y_, f_c, z_hat, a_hat


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(32, 32, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(32, 48, kernel_size=(3,3), stride=1, padding=(1,1))
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(192, 48, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 48, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 112, kernel_size=3, stride=1, padding=1),
            BasicConv2d(112, 128, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(512, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 112, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(112, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 96, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(96, 112, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(112, 112, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(112, 128, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(512, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(512, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 160, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(160, 160, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(768, 128, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(768, 192, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(192, 128, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(192, 128, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(768, 192, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(192, 224, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(224, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(256, 128, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(256, 128, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(768, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, input_channel):
        super(InceptionV4, self).__init__()
        # # Special attributs
        # self.input_space = None
        # self.input_size = (299, 299, 3)
        # self.mean = None
        # self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        # self.last_linear = nn.Linear(1536, num_classes)

    # def logits(self, features):
    #     #Allows image of any size to be processed
    #     adaptiveAvgPoolWidth = features.shape[2]
    #     x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
    #     x = x.view(x.size(0), -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, input):
        x = self.features(input)
        return x


class DP(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=768, depth=77, gamma=4):
        super(DP, self).__init__()
        self.gamma = gamma
        block = []
        for i in range(gamma + 1):
            block.append(nn.Conv2d(in_channel, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
        self.block = nn.ModuleList(block)

    def forward(self, feature):
        for i, block in enumerate(self.block):
            if i == 0:
                output = block(feature)
            else:
                output = torch.cat([output, block(feature)], dim=1)
        return output


class Descriptor(nn.Module):
    def __init__(self, input_channel=3, gamma=4):
        super(Descriptor, self).__init__()
        self.backbone = InceptionV4(input_channel)
        self.DP = DP(gamma=gamma)

    def forward(self, img):
        feature = self.backbone(img)
        f = self.DP(feature)
        return f


class TR(nn.Module):
    # translucency recovery(TR) module
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.R_t = R_t(385, beta)

    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        y_, f_c, z_hat, a = self.R_t(x, f_t, **kwargs)
        return y_, f_c, z_hat, a


class TR_new(nn.Module):
    # A new translucency recovery(TR) module with two descriptors
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_new, self).__init__()
        self.D_t_1 = Descriptor(input_channel, gamma)
        self.D_t_2 = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.AE = Pyramid_maxout(385, 3, beta)

    def forward(self, x, **kwargs):
        f_t_1 = self.D_t_1(x)
        z_hat = self.SE(f_t_1)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        z_hat_ = z_hat.detach()
        f_t_2 = self.D_t_2(x)
        a = self.AE(f_t_2)
        # yield estimated snow-free image y'
        y_ = (z_hat_ < 1) * (x - a * z_hat_) / (1 - z_hat_ + 1e-8) + (z_hat_ == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat_, a], dim=1)
        return y_, f_c, z_hat, a

class TR_za(nn.Module):
    # A  translucency recovery(TR) module predict z\times a
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_za, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.SAE = Pyramid_maxout(385, 3, beta)

    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        z_hat = self.SE(f_t)
        za = self.SAE(f_t)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        za[za >= 1] = 1
        za[za <= 0] = 0
        # yield estimated snow-free image y'
        y_ = (z_hat < 1) * (x - za) / (1 - z_hat + 1e-8) + (z_hat == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat, za], dim=1)
        return y_, f_c, z_hat, za

class RG(nn.Module):
    # the residual generation (RG) module
    def __init__(self, input_channel=7, beta=4, gamma=4):
        super(RG, self).__init__()
        self.D_r = Descriptor(input_channel, gamma)
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(385, 3, 2 * i + 1, 1, padding=i))
        self.conv_module = nn.ModuleList(block)
        self.activation = nn.Tanh()

    def forward(self, f_c):
        f_r = self.D_r(f_c)
        for i, module in enumerate(self.conv_module):
            if i == 0:
                r = module(f_r)
            else:
                r += r + module(f_r)
        r = self.activation(r)
        return r


class DesnowNet(nn.Module):
    # the DesnowNet
    def __init__(self, input_channel=3, beta=4, gamma=4, mode='original'):
        super(DesnowNet, self).__init__()
        if mode == 'original':
            self.TR = TR(input_channel, beta, gamma)
        elif mode == 'new_descriptor':
            self.TR = TR_new(input_channel, beta, gamma)
        elif mode == 'za':
            self.TR = TR_za(input_channel, beta, gamma)
        else:
            raise ValueError("Invalid architectural mode")
        self.RG = RG(beta=beta, gamma=gamma)

    def forward(self, x, **kwargs):
        y_, f_c, z_hat, a = self.TR(x, **kwargs)
        r = self.RG(f_c)
        y_hat = r + y_
        return y_hat, y_, z_hat, a


if __name__ == '__main__':

    model = DesnowNet()
    # from torchsummary import summary
    # summary(model, (3, 64, 64), device='cuda')

    from thop import profile
    input = torch.randn(1, 3, 64, 64)
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)