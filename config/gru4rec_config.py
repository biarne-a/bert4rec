import copy

from config.config import ModelConfig


class Gru4RecConfig(ModelConfig):
    def __init__(
        self,
        hidden_size=768,
        inner_dim=3072,
    ):
        """
        Builds a BertConfig.

        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param inner_dim: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        """
        self.hidden_size = hidden_size
        self.inner_dim = inner_dim

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return Gru4RecConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def get_train_features_description(self):
        from gru4rec.dataset_helpers import get_features_description
        return get_features_description()

    def get_val_and_test_features_description(self):
        from gru4rec.dataset_helpers import get_features_description
        return get_features_description()

    def get_setup_batch_fn(self, movie_id_lookup):
        from gru4rec.dataset_helpers import get_setup_batch_fn
        return get_setup_batch_fn(movie_id_lookup)

    def build_model(self, data):
        from gru4rec.gru4rec_model import Gru4RecModel
        return Gru4RecModel(data.vocab_size, **self.to_dict())

    @property
    def label_column(self):
        return "label"
