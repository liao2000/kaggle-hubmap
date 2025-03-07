# -*- coding: utf-8 -*-
import pretrainedmodels
from torchsummary import summary
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
package_dir = "./pretrained-models.pytorch-master"
sys.path.insert(0, package_dir)


def conv3x3(in_channel, out_channel, groups=1):  # not change resolusion
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=groups, bias=False)


def conv1x1(in_channel, out_channel):  # not change resolution
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

# https://github.com/moskomule/senet.pytorch/blob/8cb2669fec6fa344481726f9199aa611f08c3fbd/senet/se_module.py#L4
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = torch.sigmoid(x1 + x2)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2, 1).apply(init_weight)

    def forward(self, inputs):
        x1, _ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = torch.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(
                scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = conv1x1(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.relu(self.bn2(x)))
        x = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs))  # shortcut
        return x

class DecodeBlockLikeSEResNeXtV1(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=False):
        super().__init__()
        self.mid_channel = out_channel*2
        self.conv1x1_1 = conv1x1(
            in_channel, self.mid_channel).apply(init_weight)
        self.bn1 = nn.BatchNorm2d(self.mid_channel).apply(init_weight)
        if upsample:
            self.conv3x3 = nn.ConvTranspose2d(self.mid_channel, self.mid_channel, kernel_size=3,
                                              stride=2, output_padding=1, padding=1, dilation=1, groups=32, bias=False).apply(init_weight)
        else:
            self.conv3x3 = conv3x3(
                self.mid_channel, self.mid_channel).apply(init_weight)

        self.bn2 = nn.BatchNorm2d(self.mid_channel).apply(init_weight)
        self.conv1x1_2 = conv1x1(
            self.mid_channel, out_channel).apply(init_weight)
        self.bn3 = nn.BatchNorm2d(out_channel).apply(init_weight)

        if upsample:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, groups=32, bias=False).apply(init_weight),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = conv1x1(
                in_channel, out_channel).apply(init_weight)

        self.seBlock = SELayer(out_channel)
        self.cbam = CBAM(out_channel, reduction=16)

    def forward(self, inputs):
        x = self.conv1x1_1(inputs)
        x = F.relu(self.bn1(x))
        x = self.conv3x3(x)
        x = F.relu(self.bn2(x))
        x = self.conv1x1_2(x)
        x = F.relu(self.bn3(x))
        x = self.seBlock(x)
        x = self.cbam(x)
        shortcut = self.shortcut(inputs)
        return F.relu(x + shortcut)

class DecodeBlockLikeSEResNeXtV2(nn.Module):
    # modified from DecodeBlockLikeSEResNeXtV1
    # delete some redundant conv1x1 layers
    def __init__(self, in_channel, out_channel, upsample=False):
        super().__init__()
        self.mid_channel = out_channel*2
        if in_channel == self.mid_channel:
            self.conv1x1_1 = nn.Sequential()
        else:
            self.conv1x1_1 = conv1x1(in_channel, self.mid_channel).apply(init_weight)
            
        if upsample:
            self.conv3x3 = nn.ConvTranspose2d(self.mid_channel, self.mid_channel, kernel_size=3,
                                              stride=2, output_padding=1, padding=1, dilation=1, groups=32, bias=False).apply(init_weight)
        else:
            self.conv3x3 = conv3x3(
                self.mid_channel, self.mid_channel).apply(init_weight)

        self.bn1 = nn.BatchNorm2d(self.mid_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(self.mid_channel).apply(init_weight)
        self.conv1x1_2 = conv1x1(self.mid_channel, out_channel).apply(init_weight)
        self.bn3 = nn.BatchNorm2d(out_channel).apply(init_weight)

        if upsample:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, groups=32, bias=False).apply(init_weight),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = conv1x1(in_channel, out_channel).apply(init_weight)

        self.seBlock = SELayer(out_channel)
        self.cbam = CBAM(out_channel, reduction=16)

    def forward(self, inputs):
        x = self.conv1x1_1(inputs)
        x = F.relu(self.bn1(x))
        x = self.conv3x3(x)
        x = F.relu(self.bn2(x))
        x = self.conv1x1_2(x)
        x = F.relu(self.bn3(x))
        x = self.seBlock(x)
        x = self.cbam(x)
        shortcut = self.shortcut(inputs)
        return F.relu(x + shortcut)

# U-Net SeResNext101 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT101(nn.Module):
    def __init__(self, deepsupervision, clfhead, clf_threshold, load_weights=True):
        super().__init__()
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.clf_threshold = clf_threshold

        # encoder
        model_name = 'se_resnext101_32x4d'
        if load_weights:
            seresnext101 = pretrainedmodels.__dict__[
                model_name](pretrained='imagenet')
        else:
            seresnext101 = pretrainedmodels.__dict__[
                model_name](pretrained=None)

        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1,  # (*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool,  # ->(*,64,h/4,w/4)
            seresnext101.layer1  # ->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2  # ->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3  # ->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4  # ->(*,2048,h/32,w/32)

        # center
        self.center = CenterBlock(2048, 512)  # ->(*,512,h/32,w/32)

        # decoder
        self.decoder4 = DecodeBlock(
            512+2048, 64, upsample=True)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(
            64+1024, 64, upsample=True)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(
            64+512, 64,  upsample=True)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(
            64+256, 64,   upsample=True)  # ->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)  # ->(*,64,h,w)

        # upsample
        self.upsample4 = nn.Upsample(
            scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # deep supervision
        self.deep4 = conv1x1(64, 1).apply(init_weight)
        self.deep3 = conv1x1(64, 1).apply(init_weight)
        self.deep2 = conv1x1(64, 1).apply(init_weight)
        self.deep1 = conv1x1(64, 1).apply(init_weight)

        # final conv
        self.final_conv = nn.Sequential(
            conv3x3(320, 64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, 1).apply(init_weight)
        )

        # clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048, 512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512, 1).apply(init_weight)
        )

    def forward(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # clf head
        logits_clf = self.clf(self.avgpool(
            x4).squeeze(-1).squeeze(-1))  # ->(*,1)
        if (not self.training) & (self.clf_threshold is not None):
            if (torch.sigmoid(logits_clf) > self.clf_threshold).sum().item() == 0:
                bs, _, h, w = inputs.shape
                logits = torch.zeros((bs, 1, h, w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits, _, _
                    else:
                        return logits, _
                else:
                    if self.deepsupervision:
                        return logits, _
                    else:
                        return logits

        # center
        y5 = self.center(x4)  # ->(*,320,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        y0 = self.decoder0(y1)  # ->(*,64,h,w)

        # hypercolumns
        y4 = self.upsample4(y4)  # ->(*,64,h,w)
        y3 = self.upsample3(y3)  # ->(*,64,h,w)
        y2 = self.upsample2(y2)  # ->(*,64,h,w)
        y1 = self.upsample1(y1)  # ->(*,64,h,w)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps
            else:
                return logits

class UNET_SERESNEXT101v(nn.Module):
    def __init__(self, version, deepsupervision, clfhead, clf_threshold, load_weights=True):
        super().__init__()
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.clf_threshold = clf_threshold

        # encoder
        model_name = 'se_resnext101_32x4d'
        if load_weights:
            seresnext101 = pretrainedmodels.__dict__[
                model_name](pretrained='imagenet')
        else:
            seresnext101 = pretrainedmodels.__dict__[
                model_name](pretrained=None)

        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1,  # (*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool,  # ->(*,64,h/4,w/4)
            seresnext101.layer1  # ->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2  # ->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3  # ->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4  # ->(*,2048,h/32,w/32)

        # center
        self.center = CenterBlock(2048, 512)  # ->(*,512,h/32,w/32)

        # decoder
        if version == '1':
            self.decoder4 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV1(512+2048, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV1(128, 64)
            ) 
            self.decoder3 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV1(64+1024, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV1(128, 64)
            )
            self.decoder2 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV1(64+512, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV1(128, 64)
            )
            self.decoder1 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV1(64+256, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV1(128, 64)
            )
            self.decoder0 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV1(64, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV1(128, 64)
            )
        elif version == '2':
            # modified from v7_2
            # delete some redundant conv1x1 layers
            self.decoder4 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV2(512+2048, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV2(128, 64)
            )
            self.decoder3 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV2(64+1024, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV2(128, 64)
            )
            self.decoder2 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV2(64+512, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV2(128, 64)
            )
            self.decoder1 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV2(64+256, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV2(128, 64)
            )
            self.decoder0 = nn.Sequential(
                DecodeBlockLikeSEResNeXtV2(64, 128, upsample=True),
                DecodeBlockLikeSEResNeXtV2(128, 64)
            )
            
        # upsample
        self.upsample4 = nn.Upsample(
            scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # deep supervision
        self.deep4 = conv1x1(64, 1).apply(init_weight)
        self.deep3 = conv1x1(64, 1).apply(init_weight)
        self.deep2 = conv1x1(64, 1).apply(init_weight)
        self.deep1 = conv1x1(64, 1).apply(init_weight)

        # final conv
        self.final_conv = nn.Sequential(
            conv3x3(320, 64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, 1).apply(init_weight)
        )

        # clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048, 512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512, 1).apply(init_weight)
        )

    def forward(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # clf head
        logits_clf = self.clf(self.avgpool(
            x4).squeeze(-1).squeeze(-1))  # ->(*,1)
        if (not self.training) & (self.clf_threshold is not None):
            if (torch.sigmoid(logits_clf) > self.clf_threshold).sum().item() == 0:
                bs, _, h, w = inputs.shape
                logits = torch.zeros((bs, 1, h, w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits, _, _
                    else:
                        return logits, _
                else:
                    if self.deepsupervision:
                        return logits, _
                    else:
                        return logits

        # center
        y5 = self.center(x4)  # ->(*,320,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        y0 = self.decoder0(y1)  # ->(*,64,h,w)

        # hypercolumns
        y4 = self.upsample4(y4)  # ->(*,64,h,w)
        y3 = self.upsample3(y3)  # ->(*,64,h,w)
        y2 = self.upsample2(y2)  # ->(*,64,h,w)
        y1 = self.upsample1(y1)  # ->(*,64,h,w)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps
            else:
                return logits

def build_model(model_name, resolution, deepsupervision, clfhead, clf_threshold, load_weights):
    if model_name=='seresnext101':
        print("original model param: 134,136,011")
        model = UNET_SERESNEXT101(deepsupervision, clfhead, clf_threshold, load_weights)
    elif model_name == 'seresnext101_v1':
        print("seresnext101_v1 param: 60,134,821")
        model = UNET_SERESNEXT101v('1', deepsupervision, clfhead, clf_threshold, load_weights)
    elif model_name == 'seresnext101_v2':
        print("seresnext101_v2 param: 60,052,901")
        model = UNET_SERESNEXT101v('2', deepsupervision, clfhead, clf_threshold, load_weights)
      
    return model

if __name__ == '__main__':
    device = "cpu"
    model = build_model(model_name='seresnext101',
                        resolution=(256, 256),
                        deepsupervision=False,
                        clfhead=False,
                        clf_threshold=None,
                        load_weights=False).to(device, torch.float32)
    summary(model, (3, 256, 256), device=device)

    model = build_model(model_name='seresnext101_v1',
                        resolution=(256, 256),
                        deepsupervision=False,
                        clfhead=False,
                        clf_threshold=None,
                        load_weights=False).to(device, torch.float32)
    summary(model, (3, 256, 256), device=device)

    model = build_model(model_name='seresnext101_v2',
                        resolution=(256, 256),
                        deepsupervision=False,
                        clfhead=False,
                        clf_threshold=None,
                        load_weights=False).to(device, torch.float32)
    summary(model, (3, 256, 256), device=device)
