import tensorflow as tf


class MaskedMeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, k: int, pad_token: int = 0, name="masked_map", **kwargs):
        super().__init__(name=f"{name}_at_{k}", **kwargs)
        self._cumulative_avg_precision = tf.Variable(0.0)
        self._sum_weights = tf.Variable(0.0)
        self._k = k
        self._pad_token = pad_token
        self._k_multiplier = tf.constant([1 / curr_k for curr_k in range(1, k + 1)], dtype=tf.float32)

    def get_config(self):
        super_config = super().get_config()
        return {"k": self._k, "pad_token": self._pad_token, **super_config}

    def _average_precision_at_k(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.expand_dims(y_true, axis=-1)
        top_indices = tf.math.top_k(y_pred, k=self._k).indices
        match = y_true == top_indices
        unweighted_precisions = tf.cast(match, dtype=tf.float32)
        return tf.reduce_max(unweighted_precisions * self._k_multiplier, axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.int32)
        avg_prec_k = self._average_precision_at_k(y_true, y_pred)

        if sample_weight is None:
            sample_weight = tf.ones_like(avg_prec_k, dtype=tf.float32)
        avg_prec_k = avg_prec_k * sample_weight

        self._sum_weights.assign_add(tf.reduce_sum(sample_weight))
        self._cumulative_avg_precision.assign_add(tf.reduce_sum(avg_prec_k))

    def result(self):
        """mean average precision is the mean of the average precision for each batch"""
        return tf.math.divide_no_nan(self._cumulative_avg_precision, self._sum_weights)

    def reset_state(self):
        self._cumulative_avg_precision.assign(0.0)
        self._sum_weights.assign(0.0)
