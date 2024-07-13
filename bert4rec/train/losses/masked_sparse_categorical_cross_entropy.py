import tensorflow as tf


class MaskedSparseCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pad_token: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._pad_token = pad_token
        self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def get_config(self):
        super_config = super().get_config()
        return {"pad_token": self._pad_token, **super_config}

    def call(self, y_true, y_pred):
        # mask = y_true != self._pad_token
        loss = self._loss_object(y_true, y_pred)
        return loss

        # mask = tf.cast(mask, dtype=loss.dtype)
        # loss = tf.boolean_mask(loss, mask)

        # loss = loss * weights
        # loss = tf.reduce_sum(loss) / (tf.reduce_sum(weights) + 1e-5)
        # return loss
