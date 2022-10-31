from omegaconf import DictConfig, OmegaConf
import hydra

        
@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def test_app(cfg) -> None:
    print(cfg.trainer)
    trainer = hydra.utils.instantiate(cfg.trainer)
    print(trainer.optimizer.lr)



if __name__ == "__main__":
    import hydra
    import omegaconf

    cfg = omegaconf.OmegaConf.load("conf/config.yaml")
    print(cfg)
    test_app(cfg)

            ''' 
        in_channels,
        hidden,
        n_graph_iters,
        nb_node_layer,
        nb_edge_layer,
        emb_channels,
        cell_channels,
        layernorm,
        hidden_activation, 
        regime, 
        edge_cut,
        warmup
        '''
    
        '''
        self.in_channels = in_channels
        self.hidden = hidden
        self.n_graph_iters = n_graph_iters
        self.nb_node_layer = nb_node_layer
        self.nb_edge_layer = nb_edge_layer
        self.emb_channels = emb_channels
        self.cell_channels = cell_channels
        self.layernorm = layernorm
        self.hidden_activation = hidden_activation
        self.regime = regime
        self.edge_cut = edge_cut
        self.warmup = warmup
        # Setup input network
        self.node_encoder = make_mlp(
            self.in_channels,
            [self.hidden] * self.nb_node_layer,
            output_activation=self.hidden_activation,
            layer_norm=self.layernorm,
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * (self.hidden),
            [self.hidden] * self.nb_edge_layer + [1],
            layer_norm=self.layernorm,
            output_activation=None,
            hidden_activation=self.hidden_activation,
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            (self.hidden) * 2,
            [self.hidden] * self.nb_node_layer,
            layer_norm=self.layernorm,
            hidden_activation=self.hidden_activation,
        )

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

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.scheduler.patience,
                    gamma=self.scheduler.factor,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
        '''
