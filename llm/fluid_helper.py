import functools
import logging
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
import wandb
from fluidml.storage import LocalFileStore, TypeInfo
from lightning.fabric.utilities.cloud_io import _atomic_save as atomic_save
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.cloud_io import get_filesystem
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class MyLocalFileStore(LocalFileStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir=base_dir)

        self.type_registry["torch"] = TypeInfo(torch.save, torch.load, "pt", is_binary=True)
        self.type_registry["pl_checkpoint"] = TypeInfo(
            self._save_pl_checkpoint,
            self._load_pl_checkpoint,
            "ckpt",
            is_binary=True,
            needs_path=True,
        )
        self.type_registry["tokenizer"] = TypeInfo(self._save_tokenizer, self._load_tokenizer, needs_path=True)
        self.type_registry["pandas_dataframe"] = TypeInfo(
            self._save_pandas_dataframe,
            self._load_pandas_dataframe,
            needs_path=True,
            extension="csv",
        )

        self.type_registry["text"] = TypeInfo(self._write, self._read, extension="txt")

    @staticmethod
    def _write(obj: str, file: IO):
        file.write(obj)

    @staticmethod
    def _read(file: IO) -> str:
        return file.read()

    @staticmethod
    def _save_tokenizer(obj: PreTrainedTokenizerFast, path: str):
        obj.save_pretrained(save_directory=path, legacy_format=False)

    @staticmethod
    def _load_tokenizer(path: str) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(path)

    @staticmethod
    def _save_pandas_dataframe(obj: pd.DataFrame, path: str):
        obj.to_csv(path)

    @staticmethod
    def _load_pandas_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _save_pl_checkpoint(checkpoint: Dict[str, Any], path: Union[str, Path]):
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_save(checkpoint, path)

    @staticmethod
    def _load_pl_checkpoint(
        path: Union[str, Path],
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")
        return pl_load(path, map_location=map_location)

    def delete_run(self, task_name: str, task_unique_config: Dict):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            logger.warning(f'No run directory for task "{task_name}" and the provided unique_config exists.')
            return None

        # try to retrieve wandb synced run and delete it via api so that it is removed in dashboard
        wandb_api_path = self.load("wandb_api_path", task_name=task_name, task_unique_config=task_unique_config)
        if wandb_api_path is not None:
            # delete corresponding wandb run via api
            api = wandb.Api()
            try:
                run = api.run(wandb_api_path["wandb_api_path"])
                run.delete()
            except wandb.errors.CommError:
                logger.info(f'WandB Run {wandb_api_path["wandb_api_path"]} does not exist in WandB Server.')

        # delete retrieved run dir
        shutil.rmtree(run_dir)


def get_available_devices(use_cuda: bool = True, cuda_ids: Optional[List[int]] = None) -> List[str]:
    if (use_cuda or cuda_ids) and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f"cuda:{id_}" for id_ in cuda_ids]
        else:
            devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    return devices


def get_balanced_devices(
    count: Optional[int] = None,
    use_cuda: bool = True,
    cuda_ids: Optional[List[int]] = None,
    devices: Optional[List[str]] = None,
) -> List[str]:
    if devices is None:
        devices = get_available_devices(use_cuda, cuda_ids)

    count = count if count is not None else multiprocessing.cpu_count()
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def add_file_handler(
    log_dir: str,
    name: str = "logs",
    type_: str = "txt",
    level: Union[str, int] = "INFO",
):
    if level not in [
        "DEBUG",
        "INFO",
        "WARNING",
        "WARN",
        "ERROR",
        "FATAL",
        "CRITICAL",
        10,
        20,
        30,
        40,
        50,
    ]:
        raise ValueError(f'Logging level "{level}" is not supported.')

    log_path = os.path.join(log_dir, f"{name}.{type_}")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(processName)s - %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger_ = logging.getLogger()
    logger_.addHandler(file_handler)


def remove_file_handler():
    logger_ = logging.getLogger()
    logger_.handlers = [h for h in logger_.handlers if not isinstance(h, logging.FileHandler)]


def log_to_file(func):
    """Decorator to enable file logging for fluid ml tasks"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        run_dir = self.get_store_context().run_dir
        logger.info(f"Current run dir: {run_dir}")
        add_file_handler(run_dir)
        result = func(self, *args, **kwargs)
        remove_file_handler()
        return result

    return wrapper


@dataclass
class TaskResource:
    device: str
