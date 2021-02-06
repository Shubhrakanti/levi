'''
Code borrowed from https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb
'''
import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
import torch.nn.functional as F


# This will adding padding so the input dimesions are preserved
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

#Define a 3x3 conv with "same" padding 
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))


'''
ResNet Parts 
'''

# Basic block = (conv_bn -> relu -> conv_bn) + skip
# all convs are 3x3 with "same" padding (defined above)
class ResNetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
        
        #Only if the output channels are downsampled do we need to 
        self.skip_connection_downsample = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_downsize_skip_connection else None 

    def forward(self, x):
        residual = x
        if self.should_downsize_skip_connection: 
            residual = self.skip_connection_downsample(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_downsize_skip_connection(self):
        return self.in_channels != self.expanded_channels


class ResNetBottleNeckBlock(ResNetBasicBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    

# A basic block repeated n times with channels going from in_channels to out_channels on the first block
# we also downsample by 2 in the first block
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], blocks_per_layer=[2,2,2,2], 
                 activation=nn.ReLU, block_type=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.layers = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=blocks_per_layer[0], activation=activation, 
                        block=block_type,  *args, **kwargs),
            *[ResNetLayer(in_channels * block_type.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block_type, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, blocks_per_layer[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = F.softmax(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.layers[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=ResNetBasicBlock, blocks_per_layer=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=ResNetBasicBlock, blocks_per_layer=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=ResNetBottleNeckBlock, blocks_per_layer=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=ResNetBottleNeckBlock, blocks_per_layer=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=ResNetBottleNeckBlock, blocks_per_layer=[3, 8, 36, 3])
