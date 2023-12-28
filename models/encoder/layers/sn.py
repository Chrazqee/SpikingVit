from typing import Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode

from data.utils.types import SnnState


class SNForMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sn1 = LIFNode(tau=2., detach_reset=True, backend='cupy', step_mode='m', store_v_seq=False)
        self.sn2 = LIFNode(tau=2., detach_reset=True, backend='cupy', step_mode='m', store_v_seq=True)

        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim)

        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim)

        # the activate point in next frame has shifts than current frame, so the offset should be learnt!
        self.v_seq_shifts = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, bias=False)
        self.v_seq_bn = nn.BatchNorm2d(num_features=dim)

        self.v_seq_shifts_lif = LIFNode(tau=2., step_mode='m', detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

    def forward(self, x: torch.tensor, pre_volt: Optional[SnnState] = None):
        if pre_volt is None:
            pre_volt = torch.zeros_like(x)
            mix = self.conv1(x + pre_volt)
        else:
            mix = self.conv1(x + pre_volt[-1])
        mix = self.bn1(mix)
        cur_output = self.sn1(mix.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        cur_output = self.conv2(cur_output + x)
        cur_output = self.bn2(cur_output)
        out = self.sn2(cur_output.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        cur_state = self.sn2.v_seq.permute(1, 0, 2, 3)
        cur_output = cur_output + out

        # learnable image shifts between cur_state(cur_volt) and next frame
        shifts = (self.v_seq_shifts(cur_state))
        shifts = self.v_seq_bn(shifts)
        shifts = self.v_seq_shifts_lif(shifts.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        cur_state = cur_state + shifts

        return cur_output, cur_state
