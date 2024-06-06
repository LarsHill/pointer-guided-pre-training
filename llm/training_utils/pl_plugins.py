import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from lightning.fabric.plugins import TorchCheckpointIO

from llm.training_utils.lightning_model_wrapper import LightningModelWrapper


class HFCheckpointIO(TorchCheckpointIO):
    """Allows you to save an additional HuggingFace Hub compatible checkpoint."""

    def __init__(self, model: LightningModelWrapper, suffix: str = "_huggingface"):
        self._model = model
        self._suffix = suffix

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        super().save_checkpoint(checkpoint, path, storage_options)
        base_path = os.path.splitext(path)[0] + self._suffix
        self._model.save_hf_checkpoint(base_path)

    def remove_checkpoint(self, path: Union[str, Path]) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint
        """
        super().remove_checkpoint(path)
        hf_path = os.path.splitext(path)[0] + self._suffix
        super().remove_checkpoint(hf_path)
