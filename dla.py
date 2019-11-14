import torch
import torch.nn as nn
import math


# 参考resnet basic block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 此conv层stride有时为2有时为1(参考resnet)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # 此conv层stride始终为1(参考resnet)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x, residual=None):
        # 一般情况下第1个block会downsample,第2个block不会downsample
        # 故第1个block的residual有赋值，第2个block的redisual为None

        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


# 定义 root level为1的树的根节点
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)  # root的kennel_size为1
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = torch.cat(x, 1)  # (batch_size, channel, height, width) 这里是在channel上做concatenate
        x = self.conv(x)
        x = self.bn(x)
        if self.residual:
            x += children[0]  # todo 为什么是children[0]?
        x = self.relu(x)
        return x


# 定义 tree
class Tree(nn.Module):
    # root_dim 是树的root的in_channels数量
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_residual=False):
        super(Tree, self).__init__()

        self.levels = levels
        self.level_root = level_root

        # root_dim(root's in_channels)默认为out_channels的两倍
        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, 1, root_dim=root_dim + out_channels, root_residual=root_residual)

        if levels == 1:
            self.root = Root(root_dim, out_channels, root_residual)

        self.downsample = None
        # 如果stride>1，则第1个block会让特征图尺寸变小，故需要对x做downsample得到residual，否则x的尺寸第二个block的特征图尺寸不等，无法直接作为residual
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)  # 第一个参数为kernel_size，第二个参数为stride，都为2则下采样2倍

        # project的作用是通过1x1的卷积，让residual的channel数与第二个block的输出channel数保持一致，从而可以直接相加
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x  # level为1时，bottom就是x下采样后的结果，即传到下一个block的residual
        residual = self.project(bottom) if self.project else bottom  # 没有project时，residual就是bottom
        if self.level_root:  # todo what this mean?
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


# DLA
class DLA(nn.Module):
    # residual_root 是否对root使用残差, 小网络为False, 大网络为True
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, return_levels=False, pool_size=7):
        super(DLA, self).__init__()
        self.levels = levels
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        self.stage0 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )

        self.stage2 = Tree(levels[2], block, channels[1], channels[2], stride=2, level_root=False,
                           root_residual=residual_root)

        self.stage3 = Tree(levels[3], block, channels[2], channels[3], stride=2, level_root=True,
                           root_residual=residual_root)

        self.stage4 = Tree(levels[4], block, channels[3], channels[4], stride=2, level_root=True,
                           root_residual=residual_root)

        self.stage5 = Tree(levels[5], block, channels[4], channels[5], stride=2, level_root=True,
                           root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(len(self.levels)):
            x = getattr(self, 'stage{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


if __name__ == '__main__':
    dla34 = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, pool_size=16)
    input_data = torch.randn(1, 3, 512, 512)
    output = dla34(input_data)
