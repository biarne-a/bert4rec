import copy

from config.config import ModelConfig


class Bert4RecConfig(ModelConfig):
    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_sequence_length: int = 200,
        inner_dim=3072,
        inner_activation="gelu",
        output_dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        nb_max_masked_ids_per_seq=20,
    ):
        """
        Builds a BertConfig.

        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param num_layers: Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        :param max_sequence_length: The maximal size of sequence propagate through the network
        :param inner_dim: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        :param inner_activation: The non-linear activation function (function or string) in the encoder and pooler.
        :param output_dropout: The dropout probability for all fully connected layers in the embeddings, encoder,
         and pooler.
        :param attention_dropout: The dropout ratio for the attention probabilities.
        :param initializer_range: The stdev of the truncated_normal_initializer for initializing all weight matrices.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.inner_dim = inner_dim
        self.inner_activation = inner_activation
        self.output_dropout = output_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.nb_max_masked_ids_per_seq = nb_max_masked_ids_per_seq

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return Bert4RecConfig(**json_object)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def get_train_features_description(self):
        from bert4rec.dataset_helpers import get_features_description

        return get_features_description(self)

    def get_val_and_test_features_description(self):
        from bert4rec.dataset_helpers import get_features_description

        return get_features_description(self, nb_max_masked_ids_per_seq=1)

    def get_setup_batch_fn(self, movie_id_lookup):
        from bert4rec.dataset_helpers import get_setup_batch_fn

        return get_setup_batch_fn(movie_id_lookup)

    def build_model(self, data):
        from bert4rec.bert4rec_model import BERT4RecModel

        bert_config = self.to_dict()
        bert_config.pop("nb_max_masked_ids_per_seq")
        return BERT4RecModel(data.vocab_size, **bert_config)

    @property
    def label_column(self):
        return "masked_lm_ids"
