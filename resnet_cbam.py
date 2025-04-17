import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50

# ===================== CBAM 模块 =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

# ===================== CBAM包装ResNet，不改结构 =====================
class ResNetWithCBAM(nn.Module):
    def __init__(self, base_model, head_planes=64, tail_planes=512, num_classes=1000):
        super(ResNetWithCBAM, self).__init__()
        self.base = base_model

        self.cbam_head = CBAM(head_planes)  # conv1 输出通道
        self.cbam_tail = CBAM(tail_planes)  # layer4 输出通道（resnet34 是 512，resnet50 是 2048）

        # 重新定义 fc 以防用户改 num_classes
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.cbam_head(x)       # 插入 CBAM 不改变结构
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.cbam_tail(x)       # 再插入 CBAM

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x

# ===================== 构造函数 =====================
def resnet34_cbam(pretrained=True, num_classes=1000):
    base = resnet34(pretrained=pretrained)
    return ResNetWithCBAM(base, head_planes=64, tail_planes=512, num_classes=num_classes)

def resnet50_cbam(pretrained=True, num_classes=1000):
    base = resnet50(pretrained=pretrained)
    return ResNetWithCBAM(base, head_planes=64, tail_planes=2048, num_classes=num_classes)
