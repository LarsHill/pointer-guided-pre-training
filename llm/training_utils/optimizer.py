import inspect
from importlib import import_module

import torch

OPTIMIZERS = {
    "adam": "torch.optim.Adam",
    "adamW": "torch.optim.AdamW",
    "adagrad": "torch.optim.Adagrad",
}

LR_SCHEDULERS = {
    "exponential": "torch.optim.lr_scheduler.ExponentialLR",
    "lin_warmup": "transformers.get_linear_schedule_with_warmup",
    "one_cycle": "torch.optim.lr_scheduler.OneCycleLR",
}


class Optimizer:
    @classmethod
    def from_config(cls, type_: str, *args, **kwargs) -> torch.optim.Optimizer:
        try:
            callable_path = OPTIMIZERS[type_]
        except KeyError:
            raise KeyError(f'Optimizer "{type_}" is not implemented.')

        module_name, class_name = callable_path.rsplit(".", maxsplit=1)
        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)


class LearningRateScheduler:
    @classmethod
    def from_config(cls, type_: str, optimizer: torch.optim.Optimizer, **kwargs):
        try:
            callable_path = LR_SCHEDULERS[type_]
        except KeyError:
            raise KeyError(f'{cls.__name__} "{type_}" is not implemented.')

        module_name, class_name = callable_path.rsplit(".", maxsplit=1)
        module = import_module(module_name)
        class_ = getattr(module, class_name)

        expected_scheduler_args = inspect.signature(class_).parameters.keys()
        scheduler_kwargs = {name: value for name, value in kwargs.items() if name in expected_scheduler_args}

        return class_(optimizer=optimizer, **scheduler_kwargs)
