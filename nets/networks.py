import torch.nn as nn
import torch

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError()
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    

class Discriminator(nn.Module):
    def __init__(self, input_ch=6):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # in: 3(6) x 256 x 256

            nn.Conv2d(input_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 32 x 32

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 16 x 16

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            # out: 1 x 13 x 13

            nn.Flatten()
            )
    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True)
        )
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.enc_conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.dec_conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.Dropout(0.5),
            nn.SiLU(inplace=True)
        )
        self.dec_conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.Dropout(0.5),
            nn.SiLU(inplace=True)
        )
        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.Dropout(0.5),
            nn.SiLU(inplace=True)
        )
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(inplace=True)
        )
        self.dec_conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        e5 = self.enc_conv5(e4)
        e6 = self.enc_conv6(e5)
        
        # bottleneck
        e7 = self.enc_conv7(e6)
        d7 = self.dec_conv7(e7)
        
        # decoder
        d6 = self.dec_conv6(torch.cat([d7, e6], dim=1))
        d5 = self.dec_conv5(torch.cat([d6, e5], dim=1))
        d4 = self.dec_conv4(torch.cat([d5, e4], dim=1))
        d3 = self.dec_conv3(torch.cat([d4, e3], dim=1))
        d2 = self.dec_conv2(torch.cat([d3, e2], dim=1))
        d1 = self.dec_conv1(torch.cat([d2, e1], dim=1))
        d0 = self.dec_conv0(torch.cat([d1, e0], dim=1))
        
        return d0
        