import os
import numpy as np
from pyexpat import model
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader, NeighborLoader

from .components.track_utils import load_dataset

class TRACKMLDataModule(LightningDataModule):
    def __init__(self, 
        data_dir: str = "",
        batch_size: int = 1,
        datatype_names: list = ["train", "val", "test","test"],
        datatype_split: list = [80,10,10, 10],
        num_workers: int = 10,
        pin_memory: bool = False
        ):
        super().__init__()
        self.save_hyperparameters()
    
    def shape(self):
        print("cfg.name:", self.hparams.data_dir)

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering

        input_subdirs = [None, None, None, None]
        input_subdirs[: len(self.hparams["datatype_names"])] = [
            os.path.join(os.path.join(self.hparams["data_dir"], "trackml"), datatype)
            for datatype in self.hparams["datatype_names"]
        ]
        self.trainset, self.valset, self.testset, self.predset = [
            load_dataset(
                input_subdir=input_subdir,
                num_events=self.hparams["datatype_split"][i],
                **self.hparams
            )
            for i, input_subdir in enumerate(input_subdirs)
        ]
        for ind in self.trainset:
            ind['train_mask'] = torch.tensor(np.full(ind.x.size()[0], True)) #np.full(ind.x.size()[0], True))

    def setup_data(self):

        self.setup(stage="fit")

    def train_dataloader(self):
        if ("trainset" not in self.__dict__.keys()) or (self.trainset is None):
            self.setup_data()

        #self.trainset=Batch.from_data_list(self.trainset)
        return DataLoader(self.trainset, batch_size=self.hparams["batch_size"], num_workers=12)
        #return NeighborLoader(self.trainset, num_neighbors=[10] * 2, batch_size=1024, num_workers=16)

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=12)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=12)
        else:
            return None
        
    def predict_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=12)
        else:
            return None
    def teardown(self, stage):
        # Used to clean-up when the run is finished  
        pass
        


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "trackml.yaml")
    print(cfg)
    cfg.data_dir = str(root / "data")
    model = hydra.utils.instantiate(cfg)
    model.shape()
