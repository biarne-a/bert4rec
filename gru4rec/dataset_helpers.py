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
        # Compute the actual lengths of sequences by summing the mask along the second dimension
        lengths = tf.reduce_sum(x["input_mask"], axis=1)

        # Flatten the input_ids and mask tensors for easy extraction of values
        flat_input_ids = tf.boolean_mask(x["input_ids"], x["input_mask"])

        # Create the ragged tensor from the flattened input_ids and the calculated lengths
        input_ids = tf.RaggedTensor.from_row_lengths(flat_input_ids, lengths)

        return {
            "label": movie_id_lookup(tf.strings.as_string(x["label"])),
            "input_ids": movie_id_lookup(tf.strings.as_string(input_ids)),
        }
    return _setup_batch
