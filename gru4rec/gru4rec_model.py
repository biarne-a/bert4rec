from typing import Dict

import tensorflow as tf
from tensorflow import keras


step_signature = [{
    "input_ids": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int64),
    "label": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
}]


class Gru4RecModel(keras.models.Model):
    def __init__(self, vocab_size: int, hidden_size: int, dropout_p_embed=0.0, dropout_p_hidden=0.0, **kwargs):
        super().__init__()
        self._movie_id_embedding = tf.keras.layers.Embedding(vocab_size + 1, hidden_size)
        self._dropout_emb = tf.keras.layers.Dropout(dropout_p_embed)
        self._gru_layer = tf.keras.layers.GRU(hidden_size, recurrent_dropout=dropout_p_hidden)

    def call(self, inputs, training=False):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        ctx_movie_emb = self._dropout_emb(ctx_movie_emb, training=training)
        hidden = self._gru_layer(ctx_movie_emb, training=training)
        logits = tf.matmul(hidden, tf.transpose(self._movie_id_embedding.embeddings))
        return logits

    @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["label"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compiled_loss(y_true, y_pred)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["label"]
        y_pred = self(inputs, training=False)

        _ = self.compiled_loss(y_true, y_pred)
        # _ = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}
