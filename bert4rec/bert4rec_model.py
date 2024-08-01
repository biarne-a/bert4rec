import tensorflow as tf

from bert4rec.layers import MaskedLM, Bert4RecEncoder


step_signature = [{
    "input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_positions": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_weights": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
}]


class BERT4RecModel(tf.keras.Model):
    def __init__(self,
                 vocab_size: int,
                 mlm_activation="gelu",
                 mlm_initializer="glorot_uniform",
                 name: str = "bert4rec",
                 **kwargs):
        super().__init__(name=name)
        self._vocab_size = vocab_size
        self._config = kwargs
        self.encoder = Bert4RecEncoder(vocab_size, **kwargs)
        _ = self.encoder(self.encoder.inputs)
        self.masked_lm = MaskedLM(
            self.encoder.get_embedding_table(),
            activation=mlm_activation,
            initializer=mlm_initializer,
            name="masked_lm_predictions"
        )

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
            loss = self.compute_loss(inputs, y_true, y_pred, training=True)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        metric_values = {m.name: m.result() for m in self.metrics}
        metric_values.pop("loss")
        metric_values["fixed_loss"] = loss
        return metric_values

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        y_pred = self(inputs, training=False)

        loss = self.compute_loss(inputs, y_true, y_pred, training=False)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)

        metric_values = {m.name: m.result() for m in self.metrics}
        metric_values.pop("loss")
        metric_values["fixed_loss"] = loss
        return metric_values

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self._vocab_size,
        })
        config.update(**self._config)
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)
