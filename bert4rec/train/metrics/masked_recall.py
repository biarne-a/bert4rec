import tensorflow as tf


class MaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, k: int, name="recall", pad_token: int = 0, **kwargs):
        super().__init__(name=f"{name}_at_{k}", **kwargs)
        self._cumulative_recall = tf.Variable(0.0)
        self._sum_weights = tf.Variable(0.0)
        self._k = k
        self._pad_token = pad_token

    def get_config(self):
        super_config = super().get_config()
        return {"k": self._k, "pad_token": self._pad_token, **super_config}

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted logits.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        """
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.expand_dims(y_true, axis=-1)
        top_indices = tf.math.top_k(y_pred, k=self._k).indices
        match = y_true == top_indices
        recall_k = tf.cast(match, dtype=tf.float32)

        mask = y_true != self._pad_token
        mask_weights = tf.cast(mask, dtype=tf.float32)

        if sample_weight is not None:
            mask_weights = sample_weight * mask_weights
        recall_k = recall_k * mask_weights

        self._sum_weights.assign_add(tf.reduce_sum(mask_weights))
        self._cumulative_recall.assign_add(tf.reduce_sum(recall_k))

    def result(self):
        return tf.math.divide_no_nan(self._cumulative_recall, self._sum_weights)

    def reset_state(self):
        self._cumulative_recall.assign(0.0)
        self._sum_weights.assign(0.0)
