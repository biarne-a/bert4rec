import json
from typing import Type


class ModelConfig:
    pass


class Config:
    SEED = 42

    def __init__(
        self,
        model_config_cls: Type[ModelConfig],
        data_dir: str,
        config_file: str,
    ):
        self.data_dir = data_dir
        bert_config = json.load(open(config_file, "r"))
        self.batch_size = bert_config.pop("batch_size")
        self.learning_rate = bert_config.pop("learning_rate")
        self.bert_config = model_config_cls.from_dict(bert_config)
