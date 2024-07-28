from typing import Dict

import tensorflow as tf


def get_features_description() \
        -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "input_ids": tf.io.FixedLenFeature([200], tf.int64),
        "input_mask": tf.io.FixedLenFeature([200], tf.int64),
    }


def get_setup_batch_fn(batch_size, movie_id_lookup: tf.keras.layers.StringLookup):
    rows = tf.range(start=0, limit=batch_size, dtype=tf.int64)
    values = tf.ones(shape=(batch_size,), dtype=tf.float32)

    def _setup_batch(x):
        input_mask = x.pop("input_mask")

        # Compute the actual lengths of sequences by summing the mask along the second dimension
        lengths = tf.reduce_sum(input_mask, axis=1)

        # We retrieve one to each length to keep the last id as the target
        lengths = lengths - 1

        # Build input mask with removal of the last element of each sequence
        indices = tf.stack([rows, lengths], axis=1)
        input_shape = (batch_size, input_mask.shape[1])
        sparse_last_index = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=input_shape)
        last_index = tf.sparse.to_dense(sparse_last_index, default_value=0)
        last_index = tf.cast(last_index, dtype=tf.int64)
        input_mask = input_mask - last_index

        # Flatten the input_ids and mask tensors for easy extraction of values
        flat_input_ids = tf.boolean_mask(x["input_ids"], input_mask)

        # Create the ragged tensor from the flattened input_ids and the calculated lengths
        input_ids = tf.RaggedTensor.from_row_lengths(flat_input_ids, lengths)

        label_mask = tf.zeros(shape=input_shape, dtype=tf.int64) + last_index
        label = tf.boolean_mask(x["input_ids"], label_mask)
        x["label"] = movie_id_lookup(tf.strings.as_string(label))
        x["input_ids"] = movie_id_lookup(tf.strings.as_string(input_ids))

        return x
    return _setup_batch
