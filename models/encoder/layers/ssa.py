import torch
import torch.nn as nn
from omegaconf import DictConfig
from spikingjelly.activation_based.neuron import LIFNode


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class SSA(nn.Module):
    """
    The lif layers is used for activate neuron.
    """
    def __init__(self,
                 dim,
                 attention_cfg: [DictConfig, None],
                 num_heads: int = 8):
        """
        :param dim: channel
        :param attention_cfg: configuration
        :param num_heads: how many heads of multi-head attention
        """
        super().__init__()
        assert isinstance(num_heads, int)

        self.dim = dim

        self.scale = dim ** -0.5

        self.num_heads = attention_cfg.get('num_heads', 8) if attention_cfg is not None else 8
        self.dim_head = dim // num_heads

        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)

        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.proj_bn = nn.BatchNorm1d(dim)

        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)
        self.v_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)

    def forward(self, x):
        N, C, H, W = x.shape
        x_for_proj = x.flatten(-2, -1)
        q, k, v = self.qkv(x_for_proj).chunk(3, dim=1)

        q = self.q_bn(q).permute(1, 0, 2)
        q = self.q_lif(q).permute(1, 0, 2)
        q = q.reshape(N, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        k = self.k_bn(k).permute(1, 0, 2)
        k = self.k_lif(k).permute(1, 0, 2)
        k = k.reshape(N, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        v = self.v_bn(v).permute(1, 0, 2)
        v = self.v_lif(v).permute(1, 0, 2)
        v = v.reshape(N, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # calculate attn
        x = self._cal_attn(q, k, v) * self.scale
        x = self.attn_lif(x.reshape(N, C, H, W).permute(1, 0, 2, 3))

        x = self.proj_bn(self.proj_conv(x.flatten(-2, -1).permute(1, 0, 2))).permute(1, 0, 2)
        x = self.proj_lif(x).reshape(N, C, H, W).contiguous()
        return x

    @staticmethod
    def _cal_attn(q, k, v):
        N, num_heads, HW, dim_head = q.shape
        if HW > dim_head:  # calculate (k.transpose @ v) first
            attn = k.transpose(-2, -1) @ v
            out = q @ attn
            return out
        attn = q @ k.transpose(-2, -1)
        out = attn @ v
        return out


if __name__ == '__main__':
    input_x = torch.randn(4, 512, 12, 20).to('cuda:0')
    ssa = SSA(512, None, 8).to('cuda:0')
    output_x = ssa(input_x)
    print(output_x.shape)
