from torch import nn
from typing import List, Dict
from .mlp import make_mlp
import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
    
class VanillaGCN(nn.Module):
    def __init__(self, 
        in_channels: int = 3,
        hidden: int = 64,
        n_graph_iters: int = 8,
        nb_node_layer: int = 3,
        nb_edge_layer: int = 3,
        emb_channels: int = 0,
        layernorm: bool = True,
        hidden_activation: str = "ReLU", 
        edge_cut: float = 0.5,
        warmup: bool = False,
        ):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.in_channels = in_channels
        self.hidden = hidden
        self.n_graph_iters = n_graph_iters
        self.nb_node_layer = nb_node_layer
        self.nb_edge_layer = nb_edge_layer
        self.emb_channels = emb_channels
        self.layernorm = layernorm
        self.hidden_activation = hidden_activation
        self.edge_cut = edge_cut
        self.warmup = warmup
        # Setup input network
        self.node_encoder = make_mlp(
            in_channels,
            [hidden] * nb_node_layer,
            output_activation=hidden_activation,
            layer_norm=layernorm,
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * (hidden),
            [hidden] * nb_edge_layer + [1],
            layer_norm=layernorm,
            output_activation=None,
            hidden_activation=hidden_activation,
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            (hidden) * 2,
            [hidden] * nb_node_layer,
            layer_norm=layernorm,
            hidden_activation=hidden_activation,
        )
        print(self.in_channels)

    def forward(self, x, edge_index):

        input_x = x

        x = self.node_encoder(x)
        #         x = F.softmax(x, dim=-1)

        start, end = edge_index

        for i in range(self.n_graph_iters):

            #             x_initial = x

            messages = scatter_add(
                x[start], end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(x[end], start, dim=0, dim_size=x.shape[0])

            node_inputs = torch.cat([x, messages], dim=-1)
            #             node_inputs = F.softmax(node_inputs, dim=-1)

            x = self.node_network(node_inputs)

        #             x = x + x_initial

        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.edge_network(edge_inputs)    

if __name__ == "__main__":
    _ = VanillaGCN()
