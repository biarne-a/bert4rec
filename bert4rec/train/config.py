import json
import copy


class BertConfig:
    """Configuration for `BertModel`."""

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
    ):
        """
        Builds a BertConfig.

        :param vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param num_hidden_layers: Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        :param hidden_act: The non-linear activation function (function or string) in the encoder and pooler.
        :param hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings, encoder,
         and pooler.
        :param attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        :param max_position_embeddings: The maximum sequence length that this model might ever be used with. Typically
        set this to something large just in case (e.g., 512 or 1024 or 2048).
        :param initializer_range: The stdev of the truncated_normal_initializer for initializing all weight matrices.
        :param batch_size: The size of a batch of samples during learning or testing
        :param max_seq_length: The maximal size of sequence propagate through the network
        :param nb_max_masked_ids_per_seq: The maximum number of masked ids to predict during training
        :param num_train_steps: The number of training steps
        :param num_warmup_steps: The number of warmup steps
        :param learning_rate: The learning rate to use for backpropagation
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

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        return BertConfig(**json_object)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        return cls.from_dict(json.load(open(json_file, "r")))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Config:
    SEED = 42

    def __init__(
        self,
        data_dir: str,
        bert_config_file: str,
        batch_size: int = 256,
        nb_max_masked_ids_per_seq: int = 20,
        learning_rate: float = 1e-4,
    ):
        self.data_dir = data_dir
        self.bert_config = BertConfig.from_json_file(bert_config_file)
        self.batch_size = batch_size
        self.nb_max_masked_ids_per_seq = nb_max_masked_ids_per_seq
        self.learning_rate = learning_rate
