from typing import Dict

import tensorflow as tf
from tensorflow import keras

from train.metrics import MaskedRecall


step_signature = [{
    "input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.bool),
    "label": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
}]


class Gru4RecModel(keras.models.Model):
    def __init__(self, vocab_size: int, hidden_size: int, dropout_p_embed=0.0, dropout_p_hidden=0.0, **kwargs):
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dropout_p_embed = dropout_p_embed
        self._dropout_p_hidden = dropout_p_hidden
        self._movie_id_embedding = tf.keras.layers.Embedding(vocab_size + 1, hidden_size)
        self._dropout_emb = tf.keras.layers.Dropout(dropout_p_embed)
        self._gru_layer = tf.keras.layers.GRU(hidden_size, recurrent_dropout=dropout_p_hidden) #, use_cudnn=False
        self._recall_metric = MaskedRecall(k=10)

    def call(self, inputs, training=False):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        ctx_movie_emb = self._dropout_emb(ctx_movie_emb, training=training)
        hidden = self._gru_layer(ctx_movie_emb, training=training) #, mask=inputs["input_mask"]
        logits = tf.matmul(hidden, tf.transpose(self._movie_id_embedding.embeddings))
        return logits

    @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["label"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compute_loss(inputs, y_true, y_pred, training=True)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y_true, y_pred)
        metric_values = {m.name: m.result() for m in self.metrics}
        metric_values.pop("loss")
        metric_values["fixed_loss"] = loss
        return metric_values

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["label"]
        y_pred = self(inputs, training=False)

        loss = self.compute_loss(inputs, y_true, y_pred, training=False)
        self.compiled_metrics.update_state(y_true, y_pred)

        metric_values = {m.name: m.result() for m in self.metrics}
        metric_values.pop("loss")
        metric_values["fixed_loss"] = loss
        return metric_values

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self._vocab_size,
            "hidden_size": self._hidden_size,
            "dropout_p_embed": self._dropout_p_embed,
            "dropout_p_hidden": self._dropout_p_hidden
        })
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)
