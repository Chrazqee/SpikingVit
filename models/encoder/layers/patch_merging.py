import torch
import torch.nn as nn


# N C H W  -> N 2*C H//2 W//2
class PatchMerging(nn.Module):
    r"""
    Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layers.  Default: nn.BatchNorm2d
    """
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(in_channels=4 * dim, out_channels=2 * dim, kernel_size=1, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H * W, C
        """
        N, C, H, W = x.shape
        if H % 2:
            H -= 1
            x = x[:, :, :, :H, :]
        if W % 2:
            W -= 1
            x = x[:, :, :, :, :W]
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.permute(0, 2, 3, 1)  # x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # [B H/2 W/2 4*C]
        x = x.permute(0, 3, 1, 2)  # [B 4*C H/2 W/2]
        x = self.norm(x)  # 在卷积的前面做 bn
        x = self.reduction(x)
        x = x.reshape(N, C * 2, H // 2, W // 2)
        return x


if __name__ == "__main__":
    input_x = torch.randn(4, 64, 384, 640).to("cuda:0")
    patch_merging = PatchMerging(64).to("cuda:0")
    output_x = patch_merging(input_x)
    print(output_x.shape)
