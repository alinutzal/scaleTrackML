from pathlib import Path

import pytest
import torch

from src.datamodules.trackml_datamodule import TRACKMLDataModule


@pytest.mark.parametrize("batch_size", [1])
def test_trackml_datamodule(batch_size):
    data_dir = "data/"

    dm = TRACKMLDataModule(data_dir=data_dir, batch_size=batch_size)
    #dm.prepare_data()

    #assert not dm.trainset and not dm.valset and not dm.testset
    assert Path(data_dir, "trackml").exists()
    assert Path(data_dir, "trackml", "train").exists()

    dm.setup('fit')
    assert dm.trainset and dm.valset and dm.testset
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.trainset) + len(dm.valset) + len(dm.testset)
    assert num_datapoints == 100

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
