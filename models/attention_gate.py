import torch.nn as nn
import torch.nn.functional as F
            
class AttentionGate(nn.Module):
    def __init__(self, in_channel, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        
        self.W = nn.Sequential(nn.Conv2d(in_channel, inter_channel, kernel_size=1, stride=1, padding=0, bias=True),
                               nn.BatchNorm2d(inter_channel))
        
        self.theta = nn.Conv2d(in_channel, inter_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)

        self.phi = nn.Conv2d(gating_channel, inter_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        
        if g.shape[2] != x.shape[2] or g.shape[3] != x.shape[3]:
            g = F.interpolate(g, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        attention_map =  x * sigm_psi_f
        W_y = self.W(attention_map)
        
        return W_y, attention_map