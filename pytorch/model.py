import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3,1,1), 
                               padding=(1,0,0),
                               bias=False) 
        self.conv2 = nn.Conv3d(planes, planes, 
                               kernel_size=(1,3,3), 
                               stride=(1,stride,stride),
                               padding=(0,1,1), 
                               bias=False)
        self.conv3 = nn.Conv3d(planes, planes*self.expansion, 
                               kernel_size=(1,1,1), 
                               bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, self.expansion*planes, 
                          kernel_size=(1,1,1), 
                          stride=(1,stride,stride), 
                          bias=False),
                nn.BatchNorm3d(self.expansion*planes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out


class I3DResNet(nn.Module):

    def __init__(self, block, layers, frame_num=32, num_classes=10):
        self.inplanes = 64
        super(I3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, 
                               kernel_size=(5,7,7), 
                               stride=(1,1,1), 
                               padding=(2,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer_inflat(block, 64, layers[0])
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], stride=2)

        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.pool3 = nn.AvgPool3d((int(frame_num/4),7,7))

        self.fc = nn.Linear(512*block.expansion, num_classes, bias=False)

    def _make_layer_inflat(self, block, planes, blocks, stride=1):
        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.pool2(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return I3DResNet(Bottleneck, [3, 4, 6, 3])
