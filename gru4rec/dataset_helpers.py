from typing import Dict

import tensorflow as tf


def get_features_description() \
        -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "input_ids": tf.io.FixedLenFeature([200], tf.int64),
        "input_mask": tf.io.FixedLenFeature([200], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }


def get_parse_sample_fn(
  features_description,
  movie_id_lookup: tf.keras.layers.StringLookup
):
    def _parse_sample(example_proto):
        x = tf.io.parse_single_example(example_proto, features_description)
        return {
            "label": movie_id_lookup(tf.strings.as_string(x["label"])),
            "input_ids": movie_id_lookup(tf.strings.as_string(x["input_ids"])),
            "input_mask": x["input_mask"]
        }
    return _parse_sample