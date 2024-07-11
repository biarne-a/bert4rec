import tensorflow as tf


class MaskedSparseCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pad_token: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._pad_token = pad_token

    def get_config(self):
        super_config = super().get_config()
        return {"pad_token": self._pad_token, **super_config}

    def call(self, y_true, y_pred):
        mask = y_true != self._pad_token
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.boolean_mask(loss, mask)

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss
