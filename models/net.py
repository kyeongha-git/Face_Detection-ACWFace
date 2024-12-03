import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from deform_conv import DeformableConv2d
import numpy as np
from torch.nn import init    
import math

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        # nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        # nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )
    
def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    
def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SCM, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//4, stride=1)
        
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        
        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        
        self.ecm_1 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.ecm_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        
        self.shuffle_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=out_channel, bias=False)
        
        self.shuffle_bn = nn.BatchNorm2d(out_channel)

    def forward(self, input):
            conv3X3 = self.conv3X3(input)
        
            conv5X5_1 = self.conv5X5_1(input)
            conv5X5 = self.conv5X5_2(conv5X5_1)
            
            conv7X7_2 = self.conv7X7_2(conv5X5_1)
            conv7X7 = self.conv7x7_3(conv7X7_2)
        
            conv_ecm_1 = self.ecm_1(conv7X7_2)
            conv_ecm_2 = self.ecm_2(conv_ecm_1)
        
            out = torch.cat([conv3X3, conv5X5, conv7X7, conv_ecm_2], dim=1)
            
            shuffle_out = self.shuffle_conv(out)
            shuffle_out = self.shuffle_bn(shuffle_out)
            
            out = F.relu(shuffle_out)
            return out
        
class WFPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(WFPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.alpha_conv1 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1)
        self.alpha_conv2 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1)
        
    def forward(self, input):
        # names = list(input.keys())
        output1 = self.output1(input[0]) #input[0] = B_0, output1 = O_1
        output2 = self.output2(input[1]) #input[1] = B_1, output2 = O_2
        output3 = self.output3(input[2]) #input[2] = B_2, output3 = O_3
        
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        alpha_1 = torch.sigmoid(self.alpha_conv1(up3))
        
        output2 = self.merge1(alpha_1 * up3 + (1-alpha_1) * output2)
        
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        alpha_2 = torch.sigmoid(self.alpha_conv2(up2))
        
        output1 = self.merge2(alpha_2 * up2 + (1-alpha_2) * output1)
        
        out = [output1, output2, output3]
        return out
        
class AdaptiveKernelSize:
    def get_kernel_size(channels):
        k = math.ceil((math.log2(channels) + 1) / 2)
        return k if k % 2 == 1 else k + 1
        
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        adaptive_kernel_size = AdaptiveKernelSize.get_kernel_size(channels)
        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=adaptive_kernel_size, stride=1, padding=adaptive_kernel_size // 2, bias=False
        )
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
        
class EDAM(nn.Module):
    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channels=channel,reduction=reduction)
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
        
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
