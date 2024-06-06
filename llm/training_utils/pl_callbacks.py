import logging
from typing import List, Optional

import lightning as pl
import markdown
import wandb
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.loggers import Logger, TensorBoardLogger

logger = logging.getLogger(__name__)


def log_text(
    key: str,
    log_string: str,
    loggers: Optional[List[Logger]] = None,
    current_epoch: int = 0,
    global_step: int = 0,
) -> None:
    for train_logger in loggers:
        if not log_string.startswith("\t"):
            log_string = fix_formatting(log_string)
        if isinstance(train_logger, pl.pytorch.loggers.TensorBoardLogger):
            log_text_tensorboard(key, log_string, train_logger, current_epoch)
        elif isinstance(train_logger, pl.pytorch.loggers.WandbLogger):
            log_text_wandb(key, log_string, current_epoch, global_step)
        else:
            logger.error(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")


def fix_formatting(log_string: str) -> str:
    """
    In markdown, we create a code block by indenting each line with a tab \t.
    Code blocks have fixed formatting, i.e. each character has the same width and
    spacing is kept.
    This makes sure that texts like classification reports are printed as they appear
    in the console.
    Parameters
    ----------
    log_string
    Returns
    -------
    formatted log_string
    """
    lines = log_string.split("\n")
    lines = ["\t" + line for line in lines]
    return "\n".join(lines)


def log_text_tensorboard(key: str, log_string: str, train_logger: TensorBoardLogger, current_epoch: int = 0):
    train_logger.experiment.add_text(key, log_string, global_step=current_epoch)


def log_text_wandb(key: str, log_string: str, current_epoch: int = 0, global_step: int = 0):
    try:
        wandb.log(
            {
                key: wandb.Html(markdown_to_html(log_string)),
                "epoch": current_epoch,
                "batch": global_step,
            }
        )
    except wandb.errors.Error as e:
        logger.warning(
            "Unable to log string with wandb. "
            "If this happens in the validation sanity check, you can ignore this message. "
            f'Error: "{str(e)}"'
        )


def markdown_to_html(markdown_string: str) -> str:
    return markdown.markdown(markdown_string)


class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ExceptionHandling(Callback):
    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        # re-raise the KeyboardInterrupt if caught (currently not raised by pytorch lightning)
        if isinstance(exception, KeyboardInterrupt):
            raise exception

        # finish wandb run with error code
        # for some reason wandb tags failed runs as "finished"
        wandb.finish(exit_code=-1)
