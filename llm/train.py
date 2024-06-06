import copy
import json
import logging
import os
import pathlib
from collections import Counter
from enum import Enum
from functools import partial
from typing import Callable, List, Optional

import lightning as pl
import numpy as np
import wandb
from fluidml import Task
from fluidml.storage.base import Sweep
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llm import project_path
from llm.data.collate import COLLATORS
from llm.data.data_iterators import DATA_ITERATORS
from llm.data.dataset import DataIterator, StreamingDataset
from llm.models.model import Model
from llm.training_utils.lightning_model_wrapper import LightningModelWrapper
from llm.training_utils.pl_callbacks import ExceptionHandling, ProgressBar
from llm.training_utils.pl_plugins import HFCheckpointIO

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    TRAIN = "train"
    TEST = "test"


def init_model_loggers(task: Task, debug: bool = False) -> list | bool:
    if debug:
        return False

    run_dir = task.get_store_context().run_dir
    unique_run_name = os.path.split(run_dir)[-1]
    run_name = unique_run_name.rsplit("-", 1)[0]

    wandb_logger = WandbLogger(project=task.project_name, name=unique_run_name, save_dir=run_dir, group=run_name)

    return [wandb_logger]


def init_model_callbacks(training_params: dict, task: Task, debug: bool = False) -> list:
    run_dir = task.get_store_context().run_dir

    callbacks = [
        ProgressBar(),
        # LearningRateMonitor(logging_interval="step"),
        ExceptionHandling(),
    ]
    if not debug:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if training_params["callbacks"]["save_top_k"] > 0:
        model_checkpoint = ModelCheckpoint(
            monitor=training_params["callbacks"]["monitor_var"],
            dirpath=os.path.join(run_dir, "models"),
            filename="best_model",
            save_top_k=training_params["callbacks"]["save_top_k"],
            verbose=True,
            save_last=True,
            mode=training_params["callbacks"]["monitor_var_mode"],
        )
        model_checkpoint.FILE_EXTENSION = ""  # handled by fluidml file store
        callbacks.append(model_checkpoint)

    return callbacks


def init_iterators(
    paths: List[str], base_dir: str, iterator_params: dict, tokenizer: PreTrainedTokenizerFast
) -> list[DataIterator]:
    iterators = []
    for path in paths:
        name = pathlib.Path(path).parts[0]
        full_path = os.path.join(base_dir, path)
        iterator_fn = partial(DATA_ITERATORS[name], tokenizer=tokenizer, path=full_path, loading_params=iterator_params)
        iterators.append(iterator_fn)
    return iterators


def init_datasets(
    dataset_params: dict, iterator_params: dict, tokenizer: PreTrainedTokenizerFast, collate_fn: Callable
) -> tuple[StreamingDataset, StreamingDataset]:
    base_dir = dataset_params.pop("base_dir")

    train_paths = dataset_params.pop("train_paths")
    eval_paths = dataset_params.pop("eval_paths")

    train_batch_size = dataset_params.pop("train_batch_size")
    eval_batch_size = dataset_params.pop("eval_batch_size")

    eval_iterator_params = copy.deepcopy(iterator_params)
    train_iterators = init_iterators(train_paths, base_dir, iterator_params, tokenizer)
    if "apply_epoch_sliding_samples" in eval_iterator_params:
        eval_iterator_params["apply_epoch_sliding_samples"] = False
    if "random_max_segments_per_sample" in eval_iterator_params:
        eval_iterator_params["random_max_segments_per_sample"] = False
    eval_iterators = init_iterators(eval_paths, base_dir, eval_iterator_params, tokenizer)

    train_dataset_params = {**dataset_params, **{"batch_size": train_batch_size}}
    eval_dataset_params = {
        **dataset_params,
        **{
            "batch_size": eval_batch_size,
            "shuffle_samples": False,
            "shuffle_documents": False,
            "apply_weighted_sampling": False,
        },
    }

    train_dataset = StreamingDataset(data_iterator=train_iterators, collate_fn=collate_fn, **train_dataset_params)
    eval_dataset = StreamingDataset(data_iterator=eval_iterators, collate_fn=collate_fn, **eval_dataset_params)
    return train_dataset, eval_dataset


def dummy_collate(x: List):
    return x[0]


def init_tokenizer(hf_model_name: str, hf_tokenizer_name: Optional[str] = None) -> PreTrainedTokenizerFast:
    tokenizer_name = hf_tokenizer_name if hf_tokenizer_name is not None else hf_model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except OSError:
        tokenizer_name = os.path.join(project_path, tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return tokenizer


def get_dataset_statistics(data_iterator: DataIterator):
    all_labels = []
    number_blobs = 0
    number_blobs_annotated = 0
    for article_samples in data_iterator():
        for sample in article_samples:
            labels: list[str] | list[list[str]] = sample.get("labels")
            if labels:
                # add number of blobs
                number_blobs += len(labels)
                # add number of annotated blobs
                number_blobs_annotated += sum(1 for label in labels if label)

                # get all labels
                # multi-label setting: labels: List[List[str]]
                if isinstance(labels[0], list):
                    for label in labels:
                        all_labels.extend(label)
                # multi-class setting: labels: List[str]
                else:
                    all_labels.extend(labels)
    label_to_idx = {label: i for i, label in enumerate(sorted(set(all_labels)))}
    label_to_support = Counter(all_labels)
    label_support = np.array([label_to_support[k] for k in label_to_idx])
    label_no_support = number_blobs_annotated - label_support
    label_weights = label_no_support / label_support

    statistics = {
        "label_map": label_to_idx,
        "label_to_support": label_to_support,
        "label_weights": label_weights.tolist(),
        "number_blobs": number_blobs,
        "number_blobs_annotated": number_blobs_annotated,
    }
    return statistics


def get_statistics(
    train_paths: list[str],
    eval_paths: list[str],
    base_dir: str,
    train_iterator: DataIterator,
    eval_iterator: DataIterator,
) -> tuple[dict, dict]:
    logger.info("Infer dataset statistics.")

    train_dataset_name = "__".join(path.replace("/", "-").rsplit(".", 1)[0] for path in train_paths)
    eval_dataset_name = "__".join(path.replace("/", "-").rsplit(".", 1)[0] for path in eval_paths)

    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_statistics_path = os.path.join(cache_dir, f"{train_dataset_name}--statistics.json")
    eval_statistics_path = os.path.join(cache_dir, f"{eval_dataset_name}--statistics.json")

    try:
        train_statistics = json.load(open(train_statistics_path, "r"))
    except FileNotFoundError:
        train_statistics = get_dataset_statistics(train_iterator)
        json.dump(train_statistics, open(train_statistics_path, "w"), indent=4)
    try:
        eval_statistics = json.load(open(eval_statistics_path, "r"))
    except FileNotFoundError:
        eval_statistics = get_dataset_statistics(eval_iterator)
        json.dump(eval_statistics, open(eval_statistics_path, "w"), indent=4)

    return train_statistics, eval_statistics


def train(
    hf_model_name: str,
    dataset_params: dict,
    iterator_params: dict,
    collator_params: dict,
    model_params: dict,
    training_params: dict,
    task: Task,
    mode: Mode = "train",
    warm_start: Optional[bool] = None,
    seed: int = 42,
    debug: bool = False,
    get_data_statistics: Optional[bool] = None,
    hf_tokenizer_name: Optional[str] = None,
    num_workers: int = 0,
):
    # set seed and device
    pl.seed_everything(seed=seed)
    device = task.resource.device

    logger.info(f"Output directory: '{task.get_store_context().run_dir}'.")
    logger.info(f"Training model on '{device}'.")
    logger.info(f"Model name '{hf_model_name}'.")
    logger.info(f"Tokenizer name '{hf_tokenizer_name}'.")

    # init tokenizer
    tokenizer = init_tokenizer(hf_model_name, hf_tokenizer_name)

    # init data collator
    name = collator_params.pop("name")
    collate_fn = COLLATORS[name](tokenizer, "cpu", **collator_params)

    # init training and evaluation dataset
    if not os.path.isabs(dataset_params["base_dir"]):
        dataset_params["base_dir"] = os.path.join(project_path, dataset_params["base_dir"])
    base_dir = dataset_params["base_dir"]
    train_paths = dataset_params["train_paths"]
    eval_paths = dataset_params["eval_paths"]
    train_dataset, eval_dataset = init_datasets(dataset_params, iterator_params, tokenizer, collate_fn)

    label_map = None
    num_labels = None
    if get_data_statistics:
        train_statistics, eval_statistics = get_statistics(
            train_paths, eval_paths, base_dir, train_dataset.data_iterator, eval_dataset.data_iterator
        )
        label_map = train_statistics["label_map"]
        loss_weights = train_statistics["label_weights"]
        num_labels = len(label_map)
        collate_fn.label_map = label_map
        model_params["num_labels"] = num_labels
        model_params["loss_weights"] = loss_weights

    # init dataloaders for multiprocessing data loader
    # batching, shuffling and collating is handled by Streaming dataset, we use the dataloader only to easily enable
    # parallel dataloading via multiprocessing. Hence, we fix the dataloader batch_size to 1 (default) and apply a dummy
    # collate fn that simply passes forward the already collated batch from the dataset
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=dummy_collate, num_workers=num_workers)
    eval_dataloader = DataLoader(dataset=eval_dataset, collate_fn=dummy_collate, num_workers=num_workers)

    # infer training tasks from collator params if the tasks attribute exist
    try:
        model_params = {**model_params, **{"tasks": collate_fn.tasks}}
    except AttributeError:
        pass

    # init model
    model = Model.from_name(hf_name=hf_model_name, **model_params).to(device)

    metric_params = training_params.get("metric_params")
    if metric_params is not None:
        metric_params["label_map"] = label_map
        metric_params["num_classes"] = num_labels
        metric_params["num_labels"] = num_labels

    # wrapping model in lightning wrapper
    model_wrapped = LightningModelWrapper(
        model=model,
        optimizer_params=training_params.pop("optimizer"),
        lr_scheduler_params=training_params.pop("lr_scheduler"),
        metric_params=metric_params,
        unique_config=task.unique_config["train"],
    ).to(device)

    try:
        devices = [int(device.split(":")[-1])]
        accelerator = "gpu"
    except ValueError:
        devices = "auto"
        accelerator = "cpu"

    loggers = init_model_loggers(task, debug)
    callbacks = init_model_callbacks(training_params, task, debug)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=loggers,
        callbacks=callbacks,
        plugins=HFCheckpointIO(model=model_wrapped),
        **training_params["trainer"],
    )

    trainer.fit(
        model=model_wrapped,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader if mode == Mode.TRAIN else None,
        ckpt_path="last" if warm_start else None,
    )

    if mode == Mode.TEST:
        metrics: dict[str, float] = trainer.test(model=model_wrapped, dataloaders=eval_dataloader)[0]
        task.save(metrics, name="test_metrics", type_="json", indent=4)

    if not debug:
        wandb.finish()


def evaluate(test_metrics: List[Sweep], task: Task):
    # aggregate metrics across runs
    metrics = {}
    for sweep in test_metrics:
        for k, v in sweep.value.items():
            if k not in metrics:
                metrics[k] = {"all": [], "mean": None, "std": None}
            metrics[k]["all"].append(v)

    # calculate mean and standard deviations across runs
    for k, v in metrics.items():
        metrics[k]["mean"] = np.mean(metrics[k]["all"])
        metrics[k]["std"] = np.std(metrics[k]["all"])

    logger.info(f"{task.info.run_name}")
    logger.info(f"{json.dumps(metrics, indent=4)}")

    # save metrics as json
    task.save(metrics, name="reduced_metrics", type_="json", indent=4)
