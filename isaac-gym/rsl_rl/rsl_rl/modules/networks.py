import torch.nn as nn


class ConvEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, channels, activation=nn.ReLU):
        super(ConvEncoder, self).__init__()
        self.mu_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            # [(64−8+0)/4]+1 = 15
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # [(15−4+0)/2]+1 = 6
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # [(6−3+0)]+1 = 4
            activation(),
            nn.Flatten(),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.mu_net(x)
        return x
