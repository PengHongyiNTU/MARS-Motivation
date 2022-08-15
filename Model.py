import re
from torch.nn import Module
from torch.nn import Conv2d,  GroupNorm
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch.nn.functional as F
from torchinfo import summary
import ResNet



def load_model(model_type, *args, **kwargs):
    if model_type == 'ConvNet2':
        return ConvNet2(*args, **kwargs)
    elif model_type == "ConvNet3":
        return ConvNet3(*args, **kwargs)
    elif model_type == "ConvNet4":
        return ConvNet4(*args, **kwargs)
    elif model_type == "ConvNet5":
        return ConvNet5(*args, **kwargs)
    # Implement ResNet 
    elif model_type == 'ResNet18':
        return ResNet.ResNet18()
    elif model_type == 'ResNet34':
        return ResNet.ResNet34()
    elif model_type == 'ResNet50':
        return ResNet.ResNet50()
    elif model_type == 'ResNet101':
        return ResNet.ResNet101()
    elif model_type == 'ResNet152':
        return ResNet.ResNet152()

    else:
        raise ValueError(f'Unknown model type: {model_type}')


    

class ConvNet2(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0, 
                 num_groups=32):
        super(ConvNet2, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
    
        # self.bn1 = BatchNorm2d(32)
        # Replacing Batch normalization with Group normalization
        # As Batch normalization has a int parameter 
        # batches_tracked, which does not support aggregation
        # we use Group normalization instead
        self.gn1 = GroupNorm(num_groups, 32)
        self.gn2 = GroupNorm(num_groups, 64)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

    def forward(self, x):
        x = self.gn1(self.conv1(x)) 
        x = self.maxpool(self.relu(x))
        x = self.gn2(self.conv2(x)) 
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
    
    
class ConvNet3(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0,
                 num_groups=32):
        super().__init__()
        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.gn1 = GroupNorm(num_groups, 32)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.gn2 = GroupNorm(num_groups, 64)
        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.gn3 = GroupNorm(num_groups, 64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2) 
        
        self.fc1 = Linear(
            (h // 2 // 2 // 2 ) * (w // 2 // 2 // 2 ) * 64,
            hidden)
        self.fc2 = Linear(hidden, class_num)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.gn3(self.conv3(x)))
        x = self.maxpool(x)
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x 
        

class ConvNet4(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0,
                 num_groups=32):
        super().__init__()
        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.gn1 = GroupNorm(num_groups, 32)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.gn2 = GroupNorm(num_groups, 64)
        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.gn3 = GroupNorm(num_groups, 64)
        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.gn4 = GroupNorm(num_groups, 128)
        
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 //2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)
        self.dropout = dropout

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.gn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.gn4(self.conv4(x)))
        x = self.maxpool(x)
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
                 dropout=.0,
                 num_groups=32):
        super(ConvNet5, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.gn1 = GroupNorm(num_groups, 32)

        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.gn2 = GroupNorm(num_groups, 64)

        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.gn3 = GroupNorm(num_groups, 64)

        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.gn4 = GroupNorm(num_groups, 128)

        self.conv5 = Conv2d(128, 128, 5, padding=2)
        self.gn5 = GroupNorm(num_groups, 128)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)
        self.dropout = dropout

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.gn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.gn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.gn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.gn5(self.conv5(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x

           
        
        
    

    
    

if __name__ == "__main__":
    conv2 = load_model('ConvNet2', in_channels=3, h=32, w=32, hidden=2048, class_num=10,  dropout=.0)
    conv3 = load_model('ConvNet3', in_channels=3, h=32, w=32, hidden=2048, class_num=10,  dropout=.0)
    conv4 = load_model('ConvNet4', in_channels=3, h=32, w=32, hidden=2048, class_num=10,  dropout=.0)
    conv5 = load_model('ConvNet5', in_channels=3, h=32, w=32, hidden=2048, class_num=10, dropout=.0)
    conv2 = conv2.cuda()
    conv3 = conv3.cuda()
    conv4 = conv4.cuda()
    conv5 = conv5.cuda()
    summary(conv2, (1, 3, 32, 32))
    summary(conv3, (1, 3, 32, 32))
    summary(conv4, (1, 3,  32, 32))
    summary(conv5, (1, 3, 32, 32))
 
    

