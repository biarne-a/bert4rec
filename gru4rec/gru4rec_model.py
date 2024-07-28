from typing import Dict

import tensorflow as tf
from tensorflow import keras


step_signature = [{
    "input_ids": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int64),
    "label": tf.TensorSpec(shape=(None,), dtype=tf.int64),
}]


class Gru4RecModel(keras.models.Model):
    def __init__(self, vocab_size: int, hidden_size: int, inner_dim: int, **kwargs):
        super().__init__()
        self._movie_id_embedding = tf.keras.layers.Embedding(vocab_size + 1, hidden_size)
        self._gru_layer = tf.keras.layers.GRU(hidden_size) #hidden_size

    def call(self, inputs, training=False):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        hidden = self._gru_layer(ctx_movie_emb)
        logits = tf.matmul(hidden, tf.transpose(self._movie_id_embedding.embeddings))
        return logits

    @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["label"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compiled_loss(y_true=y_true, y_pred=y_pred)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["label"]
        y_pred = self(inputs, training=False)

        _ = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}
