from sqlite3 import paramstyle
from typing import Any, List
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

from sklearn.metrics import roc_auc_score


class VanillaGNNLitModule(LightningModule): #GNNBase
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cell_channels: int = 0,
        weight: int = 2,
        regime: List = ["pid"],
        edge_cut: float = 0.5,
        warmup: int = 0
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.net = net
 
    def forward(self, x: torch.Tensor, edge_index):
        return self.net(x, edge_index)
        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], axis=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    def handle_directed(self, batch, edge_sample, truth_sample):

        edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
        truth_sample = truth_sample.repeat(2)

        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]

        return edge_sample, truth_sample

    def training_step(self, batch, batch_idx):
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else ((~batch.y_pid.bool()).sum() / batch.y_pid.sum()).clone().detach().requires_grad_(False)
        )

        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )

        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, truth)
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=10000)

        return loss

    def log_metrics(self, score, preds, truth, batch, loss):

        edge_positive = preds.sum().float()
        edge_true = truth.sum().float()
        edge_true_positive = (
            (truth.bool() & preds).sum().float()
        )

        eff = edge_true_positive.clone().detach() / max(1, edge_true)
        pur = edge_true_positive.clone().detach() / max(1, edge_positive)

        auc = roc_auc_score(truth.bool().cpu().detach(), score.cpu().detach())

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {
                "val/loss": loss,
                "auc": auc,
                "eff": eff,
                "pur": pur,
                "current_lr": current_lr,
            }, on_epoch=True, on_step=False, batch_size=10000
        )

    def shared_evaluation(self, batch, batch_idx, log=False):
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else ((~batch.y_pid.bool()).sum() / batch.y_pid.sum()).clone().detach().requires_grad_(True)
        )

        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )

        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, truth)
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()


        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        # Edge filter performance
        score = torch.sigmoid(output)
        preds = score > self.hparams["edge_cut"]

        if log:
            self.log_metrics(score, preds, truth_sample, batch, loss)

        return {
            "loss": loss,
            "score": score,
            "preds": preds,
            "truth": truth_sample,
        }

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)
        self.log('val/loss', outputs["loss"], batch_size=1)
    """
    def validation_epoch_end(self, outputs):
        print(outputs["loss"])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return avg_loss        
    """
    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)
        self.log('test/loss', outputs["loss"], batch_size=1)
    """
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return avg_loss
    """    
    def predict_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg["lr"]

                #optimizer.param_groups[0]["lr"] = lr_scale * optimizer.param_groups[0]["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()   

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "trackml.yaml")
    _ = hydra.utils.instantiate(cfg)
