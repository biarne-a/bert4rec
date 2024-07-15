from absl import logging
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

    @property
    def identifier(self):
        return "bert4rec"

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, list):
            logging.warning('List inputs to the Bert Model are discouraged.')
            inputs = dict([
                (ref.name, tensor) for ref, tensor in zip(self.inputs, inputs)
            ])

        outputs = dict()
        encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "input_mask": inputs["input_mask"],
        }
        encoder_network_outputs = self.encoder(encoder_inputs, training=training)
        if isinstance(encoder_network_outputs, list):
            outputs['pooled_output'] = encoder_network_outputs[1]
            # When `encoder_network` was instantiated with return_all_encoder_outputs
            # set to True, `encoder_network_outputs[0]` is a list containing
            # all transformer layers' output.
            if isinstance(encoder_network_outputs[0], list):
                outputs['encoder_outputs'] = encoder_network_outputs[0]
                outputs['sequence_output'] = encoder_network_outputs[0][-1]
            else:
                outputs['sequence_output'] = encoder_network_outputs[0]
        elif isinstance(encoder_network_outputs, dict):
            outputs = encoder_network_outputs
        else:
            raise ValueError('encoder_network\'s output should be either a list '
                             'or a dict, but got %s' % encoder_network_outputs)

        sequence_output = outputs["sequence_output"]
        # Inference may not have masked_lm_positions and mlm_logits are not needed
        if "masked_lm_positions" in inputs:
            masked_lm_positions = inputs["masked_lm_positions"]
            predicted_logits = self.masked_lm(sequence_output, masked_lm_positions)
            outputs["mlm_logits"] = predicted_logits

        return outputs

    # @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            y_pred = outputs["mlm_logits"]
            loss = self.compiled_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    # @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        """
        Custom train_step function to alter standard training behaviour

        :return:
        """
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        outputs = self(inputs, training=False)
        y_pred = outputs["mlm_logits"]

        loss = self.compiled_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)
