import pyrootutils
import numpy as np
# External imports
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datamodules.components.track_utils import get_metrics

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    all_efficiencies, all_purities = [], []
    all_cuts = np.linspace(0.0, 1.0, 11)

    with torch.no_grad():
        for cut in all_cuts:

            model.hparams.edge_cut = cut
            pred_results = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
            mean_efficiency, mean_purity = get_metrics(pred_results)

            all_efficiencies.append(mean_efficiency)
            all_purities.append(mean_purity)
    
    log.info("Results!")
    print(cfg.paths.output_dir)
    with open(Path(cfg.paths.output_dir, "results.log"), "w") as file:
        np.savetxt(file, np.around((all_cuts,all_efficiencies,all_purities), decimals=4),fmt='%.4f')
    print(np.around(all_cuts,decimals=4))
    print(np.around(all_efficiencies,decimals=4))
    print(np.around(all_purities,decimals=4))

    plt.figure(figsize=(12, 8))
    plt.plot(all_cuts, all_efficiencies, label="Efficiency")
    plt.plot(all_cuts, all_purities, label="Purity")
    plt.legend()
    plt.title("Performance", fontsize=24), plt.xlabel("Edge cut", fontsize=18), plt.ylabel(
        "Eff and Purity of GNN", fontsize=18);
    plt.savefig('gnnenp.png')   

    plt.figure(figsize=(12, 8))
    plt.plot(all_efficiencies, all_purities, label="AUC")
    plt.legend()
    plt.title("Performance", fontsize=24), plt.xlabel("Efficiency", fontsize=18), plt.ylabel(
        "Purity", fontsize=18);
    plt.savefig('gnnauc.png') 

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
