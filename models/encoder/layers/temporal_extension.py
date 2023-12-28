from os.path import join, dirname, isfile

import torch
import torch.nn as nn
from omegaconf import DictConfig
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.LeakyReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        path = join(dirname(__file__), "../kernel_init/trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        x = x[None, ..., None]

        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    # k(x, y, t) = \delta (x, y) * max(0, 1 - |\frac{t}{\Delta t}|
    # noinspection PyMethodMayBeStatic
    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


class TemporalExtension(nn.Module):
    """
    The output of Spiking Down-Sample module is so `sparse`! So it should be added some extra params!
    """
    def __init__(self, downsample_cfg: [DictConfig, None]):
        super().__init__()
        dim_in = downsample_cfg.get('input_dim', 20) if downsample_cfg is not None else 20
        assert isinstance(dim_in, int)
        dim_out = downsample_cfg.get('first_layer_dim', 32) if downsample_cfg is not None else 32
        assert isinstance(dim_out, int)

        norm_affine = downsample_cfg.get('norm_affine', True) if downsample_cfg is not None else True
        assert isinstance(norm_affine, bool)

        kernel_size = 3
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=dim_in,
                              out_channels=dim_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=2,
                              bias=False)

        self.norm = nn.BatchNorm2d(num_features=dim_out)

        # learnable positional embedding
        self.lpe_conv = nn.Conv2d(in_channels=dim_out,
                                  out_channels=dim_out,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=1,
                                  bias=False)
        self.lpe_norm = nn.BatchNorm2d(num_features=dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x_lep = self.lpe_conv(x)
        x_lep = self.lpe_norm(x_lep)
        x = x + x_lep
        return x


def nChw_2_nhwC(x: torch.Tensor):
    """N C H W -> N H W C
    """
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1)


def nhwC_2_nChw(x: torch.Tensor):
    """N H W C -> N C H W
    """
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)
