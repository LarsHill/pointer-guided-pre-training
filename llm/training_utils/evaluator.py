import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from llm.training_utils.metrics import Metric

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], aggregator: Optional[Callable] = None):
        self.metrics = metrics
        self.aggregator = aggregator

    @property
    def metric_names(self) -> List:
        return list(self.metrics.keys())

    @classmethod
    def from_config(cls, aggregator: Optional[Callable] = None, **evaluator_params):
        """Creates evaluator instance and instantiates all metrics.

        Args:
            aggregator: A callable that takes all metrics as input and outputs an updated metric dict.
            **evaluator_params: Key word args needed for metric instantiation.

        Returns:
            An evaluator instance.
        """

        label_map = evaluator_params.pop("label_map")
        metrics = {}
        for split in ["train", "valid", "test"]:
            for metric_name, metric_kwargs in evaluator_params.items():
                metric_kwargs["label_map"] = label_map
                task_names = metric_kwargs.pop("task_names")

                if isinstance(task_names, str):
                    task_names = [task_names]
                if task_names is not None:
                    for task_name in task_names:
                        full_name = f"{split}-{metric_name}--{task_name}"
                        metrics[full_name] = Metric.from_config(name=full_name, **metric_kwargs)
                else:
                    full_name = f"{split}-{metric_name}"
                    metrics[full_name] = Metric.from_config(name=full_name, **metric_kwargs)

        return cls(metrics=metrics, aggregator=aggregator)

    def update(
        self,
        batch_output: Dict[str, Any],
        split: Optional[str] = None,
    ):
        for metric_name, metric in self.metrics.items():
            if split is None or metric_name.startswith(split):
                # infer expected arguments to update metric
                expected_arguments = inspect.signature(metric.update).parameters.keys()

                # get task name if one is defined
                task_name = metric_name.split("--", 1)[-1] if "--" in metric_name else None

                # if task name exists: Strip it from actual model output keys
                metric_kwargs = {}
                if task_name:
                    metric_kwargs = {
                        k.lstrip(f"{task_name}_"): v for k, v in batch_output.items() if k.startswith(task_name)
                    }

                for arg in expected_arguments:
                    if arg not in batch_output and arg not in metric_kwargs:
                        logger.warning(f"Argument {arg} is expected by metric {metric.name} but not provided by batch.")

                # update metric_kwargs with other expected arguments
                metric_kwargs.update({arg: batch_output[arg] for arg in expected_arguments if arg in batch_output})

                # update metric
                metric.update(**metric_kwargs)

    def compute(self, reset: bool = False, split: Optional[str] = None) -> Dict:
        metrics = {}
        for metric_name, metric in self.metrics.items():
            if not split or metric_name.startswith(split):
                out = metric.compute_wrapped(reset=reset)
                if isinstance(out, Dict):
                    metrics = {**metrics, **out}
                else:
                    metrics[metric_name] = out

        # if an aggregator is provided, it will be called with all calculated metrics
        if self.aggregator is not None:
            metrics = self.aggregator(metrics, split)

        return metrics
