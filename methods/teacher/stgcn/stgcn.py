import torch
import torch.nn as nn

from .stgcn_layers import STConvBlock, OutputBlock


class STGCNChebGraphConv(nn.Module):

    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate):
        super(STGCNChebGraphConv, self).__init__()
        # [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]]
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:

        x = history_data.permute(0, 3, 1, 2).contiguous()

        x = self.st_blocks(x)
        x, out_emb = self.output(x)


        x = x.transpose(2, 3)
        return x, out_emb




