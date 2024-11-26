import numpy as np
import torch
from torch import nn
from torch.nn import init
import math

class AdaptiveKernelSize:
    def get_kernel_size(channels):
        k = math.ceil((math.log2(channels) + 1) / 2)
        return k if k % 2 == 1 else k + 1

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        adaptive_kernel_size = AdaptiveKernelSize.get_kernel_size(channels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=adaptive_kernel_size, stride=1, padding=adaptive_kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        avg_pool_1d = avg_pool.view(x.size(0), 1, -1)
        max_pool_1d = max_pool.view(x.size(0), 1, -1)

        # Adaptive 1D Convolution
        avg_out = self.sigmoid(self.conv1d(avg_pool_1d))
        max_out = self.sigmoid(self.conv1d(max_pool_1d))

        combined = avg_out + max_out
        out = self.sigmoid(combined)

        return out.view(x.size(0), -1, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True) #이게 사실 상 1*1 max pooling 역할을 함 -> keepdim을 통해 텐서 형태는 유지.
        avg_result=torch.mean(x,dim=1,keepdim=True) #이게 사실 상 1*1 avg pooling 역할을 함 
        
        result=torch.cat([max_result,avg_result],1) #이러면 이게 2차원.
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual