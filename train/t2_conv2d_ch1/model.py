import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class T2Conv2DModel_Ch1(nn.Module):
    # single channel model, masks are ignored
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                padding=1,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                padding=1,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                padding=1,
                kernel_size=3,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lineer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        # convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # lineer
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lineer(x)

        return x


def main() -> int:
    model = T2Conv2DModel_Ch1()
    stats = summary(
        model,
        input_size=(1, 1, 136, 108),
        verbose=2,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        device="cpu",
    )
    print(f"total parameters: {stats.total_params:,}")
    print(f"trainable parameters: {stats.trainable_params:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())