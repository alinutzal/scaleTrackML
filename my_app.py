import pytorch_lightning
from pytorch_lightning import LightningModule
class Optimizer(LightningModule):
    algo: str
    lr: float

    def __init__(self, algo: str, lr: float) -> None:
        self.algo = algo
        self.lr = lr
        #self.save_hyperparameters(hparams)

class Dataset:
    name: str
    path: str

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path


class Trainer():
    def __init__(self, optimizer: Optimizer, dataset: Dataset) -> None:
        #super().__init__(hparams)  # type: ignore
        self.optimizer = optimizer
        self.dataset = dataset