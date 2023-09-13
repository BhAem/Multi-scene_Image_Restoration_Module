import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

# 262144 131072 65536 65536
class myMLP(nn.Module):
    def __init__(self, channel_t, channel_h, feature_dim_o=2048, feature_dim=128):
        super(myMLP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channel_t, out_channels=channel_h, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel_h, out_channels=channel_h, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        # projection head
        self.g = nn.Sequential(nn.Linear(feature_dim_o, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))


    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        print(feature.shape)
        out = self.g(feature)
        return F.normalize(out, dim=-1)




