import logging
import os

from transformers import AutoConfig, PreTrainedModel

from llm import project_path
from llm.models.bert_segment_ordering import (
    BertForSegmentOrderPretraining,
    BertForSequentialTextClassification,
)

logger = logging.getLogger(__name__)


MODELS = {
    "sequential_text_clf": BertForSequentialTextClassification,
    "language_modeling": BertForSegmentOrderPretraining,
}


class Model:
    def __init__(self):
        super().__init__()

    @classmethod
    def from_name(
        cls,
        hf_name: str,
        name: str,
        is_pretrained: bool = True,
        **model_params,
    ) -> PreTrainedModel:
        logger.info(f"Start training from pretrained checkpoint: {is_pretrained}")
        if is_pretrained:
            try:
                model = MODELS[name].from_pretrained(hf_name, **model_params)
            except OSError:
                hf_name = os.path.join(project_path, hf_name)
                model = MODELS[name].from_pretrained(hf_name, **model_params)
        else:
            try:
                model_config, unused_kwargs = AutoConfig.from_pretrained(
                    hf_name,
                    return_unused_kwargs=True,
                    **model_params,
                )
            except OSError:
                hf_name = os.path.join(project_path, hf_name)
                model_config, unused_kwargs = AutoConfig.from_pretrained(
                    hf_name,
                    return_unused_kwargs=True,
                    **model_params,
                )
            model = MODELS[name](model_config, **unused_kwargs)
        return model
