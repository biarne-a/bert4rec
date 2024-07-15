import copy
import tensorflow as tf
from typing import Optional

from bert4rec.train.layers import MaskedLM, Bert4RecEncoder


step_signature = [{
    "input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_positions": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_weights": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
}]


class BERT4RecModel(tf.keras.Model):
    """
    NOTE: The model can only be saved, when completely initialized (when using the saving api).
    For a not further known reason (but empirically tested), saving a subclassed Keras model with a
    custom `train_step()` function throws an error when not fully initialized. In detail, this line
    `loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)` causes the error. Fully
    initialized means in this context that the given metrics and loss(es) in the `model.compile()` call are not
    built but only set/initialized. See e.g. `init__()` method of the LossesContainer object
    (wrapper for compiled_metrics property):
    https://github.com/keras-team/keras/blob/3cec735c5602a1bd9880b1b5735c5ce64a94eb76/keras/engine/compile_utils.py#L117
    """

    def __init__(self,
                 vocab_size: int,
                 customized_masked_lm: Optional[tf.keras.layers.Layer] = None,
                 mlm_activation="gelu",
                 mlm_initializer="glorot_uniform",
                 name: str = "bert4rec",
                 # special_token_ids: list[int] = SPECIAL_TOKEN_IDS,
                 **kwargs):
        """

        :param encoder:
        :param customized_masked_lm:
        :param mlm_activation:
        :param mlm_initializer:
        :param name: Name of this keras model
        :param special_token_ids: An optional list of special token ids that should be prevented from
            being predicted
        :param kwargs:
        """
        super().__init__(name=name)
        encoder = Bert4RecEncoder(vocab_size, **kwargs)
        self._config = {
            "encoder": encoder,
            "customized_masked_lm": customized_masked_lm,
            "mlm_activation": mlm_activation,
            "mlm_initializer": mlm_initializer,
            "name": name,
        }

        self.encoder = encoder
        self.vocab_size = vocab_size

        _ = self.encoder(self.encoder.inputs)

        inputs = copy.copy(encoder.inputs)

        self.masked_lm = customized_masked_lm or MaskedLM(
            self.encoder.get_embedding_table(),
            activation=mlm_activation,
            initializer=mlm_initializer,
            name="masked_lm_predictions"
        )

        masked_lm_positions = tf.keras.layers.Input(
            shape=(None,), name="masked_lm_positions", dtype=tf.int32
        )
        if isinstance(inputs, dict):
            inputs["masked_lm_positions"] = masked_lm_positions
        else:
            inputs.append(masked_lm_positions)
        self.inputs = inputs

    def call(self, inputs, training=None, mask=None):
        encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "input_mask": inputs["input_mask"],
        }
        encoder_outputs = self.encoder(encoder_inputs, training=training)
        sequence_output = encoder_outputs["sequence_output"]
        masked_lm_positions = inputs["masked_lm_positions"]
        return self.masked_lm(sequence_output, masked_lm_positions)

    @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compiled_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        """
        Custom train_step function to alter standard training behaviour

        :return:
        """
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        y_pred = self(inputs, training=False)

        loss = self.compiled_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}
