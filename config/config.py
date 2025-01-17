import json


class ModelConfig:
    def get_train_features_description(self):
        raise NotImplementedError()

    def get_val_and_test_features_description(self):
        raise NotImplementedError()

    def get_setup_batch_fn(self, movie_id_lookup):
        raise NotImplementedError()


class Config:
    SEED = 42

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        config_file: str,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        json_model_config = json.load(open(config_file, "r"))
        self.batch_size = json_model_config.pop("batch_size")
        self.learning_rate = json_model_config.pop("learning_rate")
        model_config_cls = self._fetch_class(json_model_config.pop("model_config_name"))
        self.model_config = model_config_cls.from_dict(json_model_config)

    def _fetch_class(self, class_name):
        try:
            from config.bert4rec_config import Bert4RecConfig
            from config.gru4rec_config import Gru4RecConfig

            return locals()[class_name]
        except KeyError:
            raise ValueError(f"Class '{class_name}' not found")

    @property
    def results_dir(self):
        return f"{self.data_dir}/results/{self.dataset_name}"
