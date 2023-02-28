from math import prod
import torch
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

class DiscriminatorNetwork2(nn.Module):
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
        )
        self.last_layer = nn.Sequential(
            Conv2d(512, 1, 4, 1, 0, bias=False),
            Sigmoid(),
        )
    def forward(self, x):
        x = self.main(x)
        return self.last_layer(x)
class Generator128Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def ConvTransposeBlock(in_c, out_c, kernel_size, stride, padding):
            return nn.Sequential(
                    ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
                    BatchNorm2d(out_c),
                    ReLU(True)
                    )
        self.main = nn.Sequential(
                # Input (128, 1, 1)
                ConvTranspose2d(128,1024, 4, 1, 0),
                ReLU(True),

                # (1024, 4, 4)
                ConvTransposeBlock(1024, 512, 4, 2, 1),
                # (512, 8, 8)
                ConvTransposeBlock(512, 256, 4, 2, 1),
                # (256, 16, 16)
                ConvTransposeBlock(256, 128, 4, 2, 1),
                # (128, 32, 32)
                ConvTransposeBlock(128, 64, 4, 2, 1),
                # (64, 32, 32)
                ConvTransposeBlock(64, 3, 4, 2, 1),
                # (32, 64, 64)
                # (32, 128, 128)
                nn.Tanh(),
                )
    def forward(self, x):
        x = self.main(x)
        return x

class Discriminator128Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def ConvBlock(in_c, out_c, kernel_size, stride, padding):
            return nn.Sequential(
                    Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                    BatchNorm2d(out_c),
                    LeakyReLU(0.2, True)
                    )
        self.feature_extractor = nn.Sequential(
                    # (3, 128, 128)
                    Conv2d(3, 32, 4, 2, 1, bias=False),
                    LeakyReLU(0.2, True),

                    # (32, 64, 64)
                    ConvBlock(32, 64, 4, 2, 1),
                    # (64, 32, 32)
                    ConvBlock(64, 128, 4, 2, 1),
                    # (128, 16, 16)
                    ConvBlock(128, 256, 4, 2, 1),
                    # (256, 8, 8)
                    ConvBlock(256, 512, 4, 2, 1)
                    # (512, 4, 4)
                )
        self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 4 * 4, 1024),
                    nn.LeakyReLU(0.2, True),
                    nn.Linear(1024, 1),
                    nn.Sigmoid()
                )
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)
        
