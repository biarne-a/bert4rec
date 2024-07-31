from typing import Dict

import tensorflow as tf


def get_features_description() \
        -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "input_ids": tf.io.FixedLenFeature([200], tf.int64),
        "input_mask": tf.io.FixedLenFeature([200], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }


def get_setup_batch_fn(movie_id_lookup: tf.keras.layers.StringLookup):
    def _setup_batch(x):
        return {
            "label": movie_id_lookup(tf.strings.as_string(x["label"])),
            "input_ids": movie_id_lookup(tf.strings.as_string(x["input_ids"])),
            "input_mask": tf.cast(x["input_mask"], tf.bool)
        }
    return _setup_batch
