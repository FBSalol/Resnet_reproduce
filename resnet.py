import torch
import torch.nn as nn

def conv3x3(in_channel, out_channel, stride=1):
    '''
    stride: 步长
    padding: 填充
    '''
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def norm_layer(channel):
    return nn.BatchNorm2d(channel)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        
        # 图中右边shortcut（残差）的部分
        # 论文中模型架构的虚线部分，需要下采样，保证输出与identity相加时，通道形状一致
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels: # stride不为1会导致形状不一样
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x
        # 图中左边正常的前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)
        return out
        
# Bottleneck块与BasicBlock类似
class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, int(out_channel / 4), stride=stride)
        self.bn1 = norm_layer(int(out_channel / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(int(out_channel / 4), int(out_channel / 4))
        self.bn2 = norm_layer(int(out_channel / 4))
        self.conv3 = conv1x1(int(out_channel / 4), out_channel)
        self.bn3 = norm_layer(out_channel)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                conv1x1(in_channel, out_channel, stride=stride),
                norm_layer(out_channel)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(identity)
        out = self.relu(out)
        return out
    
class ResNet_18(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=1000, zero_init_residual=False):
        super(ResNet_18, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2 ,padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, 2, stride=1)# 根据结构图，只有第一个层形状不变，后面三个层均减半，所以stride=2
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 ,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def make_layer(self, block, out_channel, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel
        for i in range (1,num_blocks):
            layers.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)# 输入全连接层前将向量转化为2维
        x = self.fc(x)

        return x
    
class ResNet_34(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=1000, zero_init_residual=False):
        super(ResNet_34, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2 ,padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, 3, stride=1)# 根据结构图，只有第一个层形状不变，后面三个层均减半，所以stride=2
        self.layer2 = self.make_layer(block, 128, 4, stride=2)
        self.layer3 = self.make_layer(block, 256, 6, stride=2)
        self.layer4 = self.make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def make_layer(self, block, out_channel, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))# 因为第一个的stride可能不是1，所以需要单独append
        self.in_channel = out_channel
        for i in range (1,num_blocks):
            layers.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)# 输入全连接层前将向量转化为2维
        x = self.fc(x)

        return x
    
class ResNet_50(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=1000, zero_init_residual=False):
        super(ResNet_50, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2 ,padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 256, 3, stride=1)# 根据结构图，只有第一个层形状不变，后面三个层均减半，所以stride=2
        self.layer2 = self.make_layer(block, 512, 4, stride=2)
        self.layer3 = self.make_layer(block, 1024, 6, stride=2)
        self.layer4 = self.make_layer(block, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def make_layer(self, block, out_channel, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))# 因为第一个的stride可能不是1，所以需要单独append
        self.in_channel = out_channel
        for i in range (1,num_blocks):
            layers.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)# 输入全连接层前将向量转化为2维
        x = self.fc(x)

        return x
    
class ResNet_101(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=1000, zero_init_residual=False):
        super(ResNet_101, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2 ,padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 256, 3, stride=1)# 根据结构图，只有第一个层形状不变，后面三个层均减半，所以stride=2
        self.layer2 = self.make_layer(block, 512, 4, stride=2)
        self.layer3 = self.make_layer(block, 1024, 23, stride=2)
        self.layer4 = self.make_layer(block, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def make_layer(self, block, out_channel, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))# 因为第一个的stride可能不是1，所以需要单独append
        self.in_channel = out_channel
        for i in range (1,num_blocks):
            layers.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)# 输入全连接层前将向量转化为2维
        x = self.fc(x)

        return x
    
class ResNet_152(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=1000, zero_init_residual=False):
        super(ResNet_152, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2 ,padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 256, 3, stride=1)# 根据结构图，只有第一个层形状不变，后面三个层均减半，所以stride=2
        self.layer2 = self.make_layer(block, 512, 8, stride=2)
        self.layer3 = self.make_layer(block, 1024, 36, stride=2)
        self.layer4 = self.make_layer(block, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def make_layer(self, block, out_channel, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))# 因为第一个的stride可能不是1，所以需要单独append
        self.in_channel = out_channel
        for i in range (1,num_blocks):
            layers.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)# 输入全连接层前将向量转化为2维
        x = self.fc(x)

        return x
    
if __name__ == '__main__':
    model = ResNet_18()
    model.load_state_dict(torch.load('resnet18.pth'))
    print(model)
    img = torch.randn(1,3,224,224)
    output = model(img)
    print(output.size())