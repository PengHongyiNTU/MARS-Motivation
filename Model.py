from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models import resnet18, resnet34, resnet50, vgg11



def load_model(model_type, *args, **kwargs):
    if model_type == 'ConvNet2':
        return ConvNet2(*args, **kwargs)
    if model_type == "ConvNet5":
        return ConvNet5(*args, **kwargs)
    if model_type == "VGG11":
        return vgg11()
    elif model_type == 'ResNet18':
        return resnet18(weights=None)
    elif model_type == 'ResNet34':
        return resnet34(weights=None)
    elif model_type == 'ResNet50':
        return resnet50(weights=None)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


    
# Normally Federated learning do not use batch normalization
class ConvNet2(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0):
        super(ConvNet2, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
    
        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

    def forward(self, x):
        x = self.bn1(self.conv1(x)) 
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) 
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
    
class ConvNet5(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0):
        super(ConvNet5, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.bn3 = BatchNorm2d(64)

        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.bn4 = BatchNorm2d(128)

        self.conv5 = Conv2d(128, 128, 5, padding=2)
        self.bn5 = BatchNorm2d(128)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x
    

    
    

if __name__ == "__main__":
    conv2 = load_model('ConvNet2', in_channels=3, h=32, w=32, hidden=2048, class_num=10,  dropout=.0)
    conv2 = conv2.cuda()
    conv5 = load_model('ConvNet5', in_channels=3, h=32, w=32, hidden=2048, class_num=10, dropout=.0)
    conv5 = conv5.cuda()
    vgg11 = load_model('VGG11')
    vgg11 = vgg11.cuda()
    summary(conv2, (16, 3, 32, 32))
    summary(conv5, (16, 3, 32, 32))
    summary(vgg11, (16, 3, 224, 224))
    
    
    