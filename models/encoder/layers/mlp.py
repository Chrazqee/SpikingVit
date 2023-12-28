import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode


class MLP(nn.Module):
    """two linear projection!
    Args:
        in_features: input channel
        hidden_features: hidden channel
        out_features: output channel
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1, bias=False)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1, bias=False)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(-2, -1)
        N, C, HW = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).permute(1, 0, 2)
        x = self.fc1_lif(x).permute(1, 0, 2)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).permute(1, 0, 2)
        x = self.fc2_lif(x).permute(1, 0, 2)

        x = x.reshape(N, C, H, W)
        return x


if __name__ == "__main__":
    input_x = torch.randn(4, 64, 96, 160).to("cuda:0")
    mlp = MLP(64, 128, 64).to("cuda:0")
    output_x = mlp(input_x)
    print(output_x.shape)
