import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, pretrain_url):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.encoder = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](False)

        if pretrained:
            cached_file = pretrain_url
            state_dict = torch.load(cached_file)
            new_state_dict = self.encoder.state_dict()

            # 删除pretrained_dict.items()中model所没有的东西
            state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数

            new_state_dict.update(state_dict)
            self.encoder.load_state_dict(new_state_dict)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


if __name__=='__main__':
    # model = ResnetEncoder(18, False, "")
    # image = torch.randn(4, 3, 192, 640)
    # output = model(image)
    # print(output)

    net = ResnetEncoder(18, False, "")
    summary(net, (3, 224, 224), device='cpu')
