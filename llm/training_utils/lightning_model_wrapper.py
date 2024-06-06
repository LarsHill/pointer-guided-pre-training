import inspect
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import lightning as pl
import torch
from transformers import PreTrainedModel

from llm.training_utils.metrics import init_metric_collection, update_metrics
from llm.training_utils.optimizer import LearningRateScheduler, Optimizer
from llm.training_utils.pl_callbacks import log_text
from llm.utils import to_device

logger = logging.getLogger(__name__)


def convert_segment_order_output_to_text(preds: torch.Tensor, labels: torch.Tensor, acc: float) -> str:
    """Converts segment order batch output in text format for logging.

    Output format is:
        Batch Acc: 24.24
        TP:     0010 | 0000 | 00100 | 10001 | 100 | 0010 | 100 | 00001
        Preds:  2222 | 2111 | 44444 | 44441 | 222 | 2222 | 222 | 44444
        Labels: 0321 | 1230 | 32410 | 40231 | 201 | 0321 | 201 | 03124
    """
    # create text logging
    tp = (preds == labels).float()
    mask = labels != -100

    tp_list = [r[m].int().tolist() for r, m in zip(tp, mask)]
    preds_list = [r[m].int().tolist() for r, m in zip(preds, mask)]
    labels_list = [r[m].int().tolist() for r, m in zip(labels, mask)]

    acc_str = f"Batch Acc: {round(acc * 100, 2)}"
    tp_str = "TP:     " + " | ".join(",".join([str(i) for i in elem]) for elem in tp_list)
    preds_str = "Preds:  " + " | ".join(",".join([str(i) for i in elem]) for elem in preds_list)
    labels_str = "Labels: " + " | ".join(",".join([str(i) for i in elem]) for elem in labels_list)

    return "\n".join([acc_str, tp_str, preds_str, labels_str])


def masked_accuracy(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy of softmax output with mask applied over values.

    Adapted from pointer-networks-pytorch by ast0414:
      https://github.com/ast0414/pointer-networks-pytorch
    """

    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean().detach().cpu()
        return accuracy


class Split(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __str__(self):
        return self.value


class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        optimizer_params: Optional[dict] = None,
        lr_scheduler_params: Optional[dict] = None,
        metric_params: Optional[dict] = None,
        unique_config: Optional[dict] = None,
    ):
        super().__init__()

        self.model = model

        # Optional: For Training only
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.unique_config = unique_config

        self.valid_global_step = 0
        self.metrics = None
        if metric_params is not None:
            metric_list = metric_params.pop("metrics")
            self.metrics = init_metric_collection(metric_list, device=model.device, **metric_params)

    def get_batch_size(self) -> int:
        try:
            batch_size = self.trainer.train_dataloader.dataset.batch_size
        except AttributeError:
            batch_size = self.trainer.test_dataloaders.dataset.batch_size
        return batch_size

    def log_metrics(self, metrics: Dict, split: Split, every: Literal["step", "epoch"] = "step", verbose: bool = False):
        global_step = self.trainer.global_step if split == Split.TRAIN else self.valid_global_step
        checkpoint_info = self.trainer.checkpoint_callback
        if checkpoint_info is not None:
            monitored_metric_name = checkpoint_info.monitor
            if monitored_metric_name in metrics:
                current_value = metrics[monitored_metric_name]
                if (
                    checkpoint_info.best_model_score is None
                    or (checkpoint_info.mode == "max" and current_value > checkpoint_info.best_model_score)
                    or (checkpoint_info.mode == "min" and current_value < checkpoint_info.best_model_score)
                ):
                    new_best_value = current_value
                else:
                    new_best_value = checkpoint_info.best_model_score
                metrics.update({f"best-{monitored_metric_name}": new_best_value})

        if every == "epoch" or global_step % self.trainer.log_every_n_steps == 0:
            if verbose:
                logger.info(
                    f"{json.dumps({k: v for k, v in metrics.items() if not isinstance(v, str)}, indent=4, default=str)}"
                )

            for k, v in metrics.items():
                if isinstance(v, str):
                    log_text(
                        key=f"{k}-{every}",
                        log_string=v,
                        loggers=self.trainer.loggers,
                        current_epoch=self.trainer.current_epoch,
                        global_step=global_step,
                    )

                else:
                    self.log(name=f"{k}-{every}", value=v, batch_size=self.get_batch_size())

    def forward(self, batch: dict) -> Dict:
        # move all arguments to device
        batch = to_device(batch, device=self.model.device)
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def on_train_batch_end(self, step_output: Dict, batch: Any, batch_idx: int) -> None:
        # update metrics
        if self.metrics:
            results = update_metrics(self.metrics[Split.TRAIN], {**batch, **step_output})
            #
            # self.log_metrics(
            #     metrics=results,
            #     split=Split.TRAIN,
            #     every="step",
            # )

    def on_train_epoch_end(self) -> None:
        pass
        # # compute metrics
        # if self.metrics:
        #     results = self.metrics[Split.TRAIN].compute()
        #
        #     self.log_metrics(
        #         metrics=results,
        #         split=Split.TRAIN,
        #         every="epoch",
        #     )
        #     self.metrics[Split.TRAIN].reset()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        # print(batch["titles"])
        # print(batch["blobs"])
        return self.forward(batch)

    def on_validation_batch_end(self, step_output: Dict, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # update metrics
        if self.metrics:
            update_metrics(self.metrics[Split.VALID], {**batch, **step_output})
            # results = update_metrics(self.metrics[Split.VALID], {**batch, **step_output})
            #
            # self.log_metrics(
            #     metrics=results,
            #     split=Split.VALID,
            #     every="step",
            # )

        self.valid_global_step += 1

    def on_validation_epoch_end(self) -> None:
        # compute metrics
        if self.metrics:
            results = self.metrics[Split.VALID].compute()

            self.log_metrics(
                metrics=results,
                split=Split.VALID,
                every="epoch",
            )
            self.metrics[Split.VALID].reset()

            train_results = self.metrics[Split.TRAIN].compute()

            self.log_metrics(
                metrics=train_results,
                split=Split.TRAIN,
                every="epoch",
            )
            self.metrics[Split.TRAIN].reset()

    def test_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def on_test_batch_end(self, step_output: Dict, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # update metrics
        if self.metrics:
            update_metrics(self.metrics[Split.TEST], {**batch, **step_output})

    def on_test_epoch_end(self) -> None:
        # compute and log metrics
        if self.metrics:
            results = self.metrics[Split.TEST].compute()

            self.log_metrics(
                metrics=results,
                split=Split.TEST,
                every="epoch",
            )
            self.metrics[Split.TEST].reset()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.model(**batch).dict()

    def _add_metrics_to_hyperparameter_table_logging(self, config: Dict, prefix: Optional[str] = None):
        for train_logger in self.loggers:
            if "metrics" in inspect.signature(train_logger.log_hyperparams).parameters:
                metrics = {name: 0.0 for name in self.evaluator.metric_names if prefix and name.startswith(prefix)}
                train_logger.log_hyperparams(config, metrics)
            else:
                train_logger.log_hyperparams(config)

    def on_train_start(self):
        self._add_metrics_to_hyperparameter_table_logging(config=self.unique_config, prefix="valid")

    def on_fit_start(self) -> None:
        self.trainer.fit_loop.setup_data()

    def on_test_start(self):
        self._add_metrics_to_hyperparameter_table_logging(config=self.unique_config, prefix="test")

    def configure_optimizers(self):
        optimizer = Optimizer.from_config(params=self.parameters(), **self.optimizer_params)

        scheduler = None
        if self.lr_scheduler_params is not None:
            self.trainer.fit_loop.setup_data()
            # calculate or retrieve total number of train steps
            if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
                train_steps = self.trainer.max_steps
            else:
                total_devices = self.trainer.num_devices * self.trainer.num_nodes
                train_batches = self.trainer.train_dataloader.dataset.num_batches // total_devices
                train_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches
            lr_warmup = self.lr_scheduler_params.pop("lr_warmup", 0.0)
            interval = self.lr_scheduler_params.pop("interval", "epoch")
            lr_scheduler = LearningRateScheduler.from_config(
                optimizer=optimizer,
                num_warmup_steps=lr_warmup * train_steps,
                num_training_steps=train_steps,
                **self.lr_scheduler_params,
            )

            scheduler = {
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "strict": False,
            }

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the model using the original HF AutoModel.

        This is useful for when you'd like to export the model to the hub.
        Args:
            path: Path to save the model to.
        """
        self.model.save_pretrained(path)
