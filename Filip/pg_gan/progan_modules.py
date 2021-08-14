import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv1d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose1d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv1d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))
        convs.append(EqualConv1d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))
        
        convs.append(nn.Dropout(0.2)) 

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat, step):
    if step !=6:
        return F.interpolate(feat, scale_factor=2, mode='linear', align_corners=False)
    else:
        return F.interpolate(feat, scale_factor=1.25, mode='linear', align_corners=False)

class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=128, pixel_norm=True, tanh=True):
        super().__init__()

        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            EqualConvTranspose1d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.1))
        
        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel//2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel//2, in_channel//4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel//4, in_channel//4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv1d(in_channel, 1, 1)
        self.to_rgb_16 = EqualConv1d(in_channel, 1, 1)
        self.to_rgb_32 = EqualConv1d(in_channel, 1, 1)
        self.to_rgb_64 = EqualConv1d(in_channel//2, 1, 1)
        self.to_rgb_128 = EqualConv1d(in_channel//4, 1, 1)
        self.to_rgb_256 = EqualConv1d(in_channel//4, 1, 1)
        
        self.max_step = 6

    def progress(self, feat, module, other_factor):
        if other_factor:
            out = F.interpolate(feat, scale_factor=1.25, mode='linear', align_corners=False)
        else:
            out = F.interpolate(feat, scale_factor=2, mode='linear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha, step):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1), step)
            out = (1-alpha)*skip_rgb + alpha*module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8, False)
        if step==1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.to_rgb_8(out_8)
        
        
        out_16 = self.progress(out_8, self.progression_16, False)
        if step==2:
            return self.output( out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha, step )
        
        out_32 = self.progress(out_16, self.progression_32, False)
        if step==3:
            return self.output( out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha, step )

        out_64 = self.progress(out_32, self.progression_64, False)
        if step==4:
            return self.output( out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha, step )
        
        out_128 = self.progress(out_64, self.progression_128, False)
        if step==5:
            return self.output( out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha, step )

        out_256 = self.progress(out_128, self.progression_256, True)
        if step==6:
            return self.output( out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha, step )


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(160, feat_dim//4, 3, 1),
                                          ConvBlock(feat_dim//4, feat_dim//2, 3, 1),
                                          ConvBlock(feat_dim//2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim+1, feat_dim , 3, 1, 4, 0)])
                                          
        
        self.from_rgb = nn.ModuleList([EqualConv1d(1, 160, 1),
                                       EqualConv1d(1, feat_dim//4, 1),
                                       EqualConv1d(1, feat_dim//2, 1),
                                       EqualConv1d(1, feat_dim, 1),
                                       EqualConv1d(1, feat_dim, 1),
                                       EqualConv1d(1, feat_dim, 1),
                                       EqualConv1d(1, feat_dim, 1)])
        
        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input_signal, step=0, alpha=-1):
        
        if(step == 6):
            input = input_signal.reshape([len(input_signal), 1, 160])
        else:
            input = input_signal.reshape([len(input_signal), 1, 16 * (2 ** (step-2))])
            
        for i in range(step, -1, -1):      
            index = self.n_layer - i - 1
            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)
            if i > 0:
                if i != 6:
                    out = F.interpolate(out, scale_factor=0.5, mode='linear', align_corners=False)
                else:
                    out = F.interpolate(out, scale_factor=0.8, mode='linear', align_corners=False) 
                    
                if i == step and 0 <= alpha < 1:
                    if i != 6:
                        skip_rgb = F.interpolate(input, scale_factor=0.5, mode='linear', align_corners=False) 
                    else:
                        skip_rgb = F.interpolate(input, scale_factor=0.8, mode='linear', align_corners=False) 
                        
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out
        out = out.squeeze(2)
        out = self.linear(out)

        return out
