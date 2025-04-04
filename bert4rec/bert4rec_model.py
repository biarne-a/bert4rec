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
                 mlm_initializer="TruncatedNormal",
                 name: str = "bert4rec",
                 **kwargs):
        super().__init__(name=name)
        self._vocab_size = vocab_size
        self._config = kwargs
        self._mlm_activation = mlm_activation
        self._mlm_initializer = mlm_initializer

    def build(self, input_shape):
        self.encoder = Bert4RecEncoder(self._vocab_size, **self._config)
        _ = self.encoder(self.encoder.inputs)
        self.masked_lm = MaskedLM(
            self.encoder.get_embedding_table(),
            activation=self._mlm_activation,
            initializer=self._mlm_initializer,
            initializer_range=self._config["initializer_range"],
            name="masked_lm_predictions"
        )

    def call(self, inputs, training=None, mask=None):
        encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "input_mask": inputs["input_mask"],
        }
        sequence_output = self.encoder(encoder_inputs, training=training)
        masked_lm_positions = inputs["masked_lm_positions"]
        return self.masked_lm(sequence_output, masked_lm_positions)

    @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compute_loss(inputs, y_true, y_pred, sample_weight=sample_weight)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self._update_metrics(y_true, y_pred, sample_weight, loss)

    def _update_metrics(self, y_true, y_pred, sample_weight, loss):
        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["masked_lm_ids"]
        sample_weight = inputs["masked_lm_weights"]
        y_pred = self(inputs, training=False)

        loss = self.compute_loss(inputs, y_true, y_pred, sample_weight=sample_weight)

        return self._update_metrics(y_true, y_pred, sample_weight, loss)

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
