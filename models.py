import torch.nn as nn
from torch.nn import ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Conv2d, Sigmoid


class GeneratorNetwork(nn.Module):
    def __init__(self) -> None:
        super(GeneratorNetwork, self).__init__()
        self.main = nn.Sequential(
            # Input Shape z -> (100, 1, 1)
            ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            BatchNorm2d(1024),
            ReLU(True),

            # (1024, 4, 4)
            ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            BatchNorm2d(512),
            ReLU(True),

            # (512, 8, 8)
            ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            ReLU(True),

            # (256, 16, 16)
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            ReLU(True),

            #(128, 32, 32)
            ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class DiscriminatorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # input shape (3, 64, 64)
            Conv2d(3, 64, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),

            # (64, 32, 32)
            Conv2d(64, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, True),

            #(128, 16, 16)
            Conv2d(128, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, True),

            #(256, 8, 8)
            Conv2d(256, 512, 4, 2, 1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2, True),

            #(512, 4, 4)
            Conv2d(512, 1, 4, 1, 0, bias=False),
            Sigmoid(),

            # (1, 1, 1)
        )
    def forward(self, x):
        return self.main(x)