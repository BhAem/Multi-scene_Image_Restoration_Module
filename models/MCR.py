import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrain=None):
        super(Vgg19, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features

        vgg = models.vgg19(pretrained=False)
        cached_file = pretrain
        state_dict = torch.load(cached_file)
        new_state_dict = vgg.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数
        new_state_dict.update(state_dict)
        vgg.load_state_dict(new_state_dict)
        vgg_pretrained_features = vgg.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class SCRLoss(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.vgg = Vgg19(pretrain=pretrain).cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contrastive

        return loss


# class HCRLoss(nn.Module):
#     def __init__(self, pretrain=None):
#         super().__init__()
#         self.vgg = Vgg19(pretrain=pretrain).cuda()
#         self.l1 = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
#
#     def forward(self, a, p, n):
#         a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
#         loss = 0
#
#         d_ap, d_an = 0, 0
#         for i in range(len(a_vgg)):
#             b, c, h, w = a_vgg[i].shape
#             d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
#             # a_vgg[i].unsqueeze(1).expand(b, b, c, h, w): a_vgg[i][0, 0] == a_vgg[i][0, 1] == a_vgg[i][0, 2]...
#             # n_vgg[i].expand(b, b, c, h, w): a_vgg[i][0] == a_vgg[i][1] == a_vgg[i][2]..., but a_vgg[i][0, 0] != a_vgg[i][0, 1]
#
#             t = a_vgg[i].unsqueeze(1).expand(b, b, c, h, w)
#             t2 = n_vgg[i].expand(b, b, c, h, w).detach()
#             d_an = self.l1(t, t2)
#             contrastive = d_ap / (d_an + 1e-7)
#             loss += self.weights[i] * contrastive
#
#         return loss


# class Resnet50(torch.nn.Module):
#     def __init__(self, requires_grad=False, pretrain=None):
#         super(Resnet50, self).__init__()
#         # vgg_pretrained_features = models.vgg19(pretrained=True).features
#
#         resnet50 = models.resnet50(pretrained=False)
#         cached_file = pretrain
#         state_dict = torch.load(cached_file)
#         new_state_dict = resnet50.state_dict()
#         state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数
#         new_state_dict.update(state_dict)
#         resnet50.load_state_dict(new_state_dict)
#         self.resnet50 = resnet50
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, x):
#         y = self.resnet50(x)
#         return F.normalize(y, dim=-1)


class Resnet50(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrain=None):
        super(Resnet50, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features

        resnet50 = models.resnet50(pretrained=False)
        cached_file = pretrain
        state_dict = torch.load(cached_file)
        new_state_dict = resnet50.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数
        new_state_dict.update(state_dict)
        resnet50.load_state_dict(new_state_dict)
        self.resnet50 = resnet50
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # y = self.resnet50(x)
        x = (x - 0.45) / 0.225
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x0 = self.resnet50.relu(x)
        x1 = self.resnet50.layer1(self.resnet50.maxpool(x0))
        x2 = self.resnet50.layer2(x1)
        x3 = self.resnet50.layer3(x2)
        x4 = self.resnet50.layer4(x3)
        return [x0, x1, x2, x3, x4]


class HCRLoss(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.resnet50 = Resnet50(pretrain=pretrain).cuda()
        self.temperature = 0.5
        self.l1 = nn.L1Loss()

    def forward(self, a, p, n):
        batch_size = a.size(0)
        a_vgg, p_vgg, n_vgg = self.resnet50(a), self.resnet50(p), self.resnet50(n)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            b, c, h, w = a_vgg[i].shape
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            t = a_vgg[i].unsqueeze(1).expand(b, b, c, h, w)
            t2 = n_vgg[i].expand(b, b, c, h, w).detach()
            d_an = self.l1(t, t2)
            contrastive = d_ap / (d_an + 1e-7)
            loss = loss + contrastive
        loss /= len(a_vgg)
        return loss

        # # [2*B, D]
        # out = torch.cat([out_1, out_2], dim=0)
        # # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # # 分子： *为对应位置相乘，也是点积
        # # compute loss
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # # [2*B]
        # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

