import torch
from torch import nn
from torch.nn import functional as F
# from models.networks_other import init_weights

class Attention_gate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(Attention_gate, self).__init__()
        # in_channels -> image channels, gating_channels -> signal channels
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            # self.inter_channels = in_channels // 2
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1

        # operation for image(feature map), down-sample
        self.theta = nn.Conv3d(self.in_channels, self.inter_channels, kernel_size=2, stride=2, padding=0)
        # operation for signal, reduce channel
        self.phi = nn.Conv3d(self.gating_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        # fusion operation(after fusion image and signal)
        self.psi = nn.Conv3d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0)
        # factor operation, get attention factor
        self.w = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels)
        )

    def forward(self, x, g):
        if x.shape[0] != g.shape[0]:
            raise ValueError('batch size is not matching for feature map and signal')

        theta_x = self.theta(x)
        # theta_x = x
        phi_g = self.phi(g)
        phi_g = F.interpolate(phi_g, size=theta_x.size()[2:], mode='trilinear', align_corners=False)

        x_g = F.relu(theta_x + phi_g, inplace=True)

        x_g = self.psi(x_g)

        x_g = torch.sigmoid(x_g)

        x_g_factor = F.interpolate(x_g, size=x.size()[2:], mode='trilinear', align_corners=False)

        y = x_g_factor.expand_as(x) * x

        weighted_y = self.w(y)
        return weighted_y, x_g_factor

if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['concatenation']

    for mode in mode_list:
        img = torch.rand(2, 16, 100, 100, 100)
        gat = torch.rand(2, 64, 4, 4, 4)
        net = Attention_gate(in_channels=16, inter_channels=16, gating_channels=64)
        out, sigma = net(img, gat)
        print(out.size())
