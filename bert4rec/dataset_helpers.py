from typing import Dict, Optional

import tensorflow as tf
import numpy as np

from config.bert4rec_config import Bert4RecConfig


def get_features_description(model_config: Bert4RecConfig, nb_max_masked_ids_per_seq: Optional[int] = None) \
        -> Dict[str, tf.io.FixedLenFeature]:
    max_seq_length = model_config.max_sequence_length
    nb_max_masked_ids_per_seq = nb_max_masked_ids_per_seq or model_config.nb_max_masked_ids_per_seq
    return {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.float32),
    }


def get_parse_sample_fn(
  features_description,
  movie_id_lookup: tf.keras.layers.StringLookup,
  training: bool
):
    def _parse_sample(example_proto):
        x = tf.io.parse_single_example(example_proto, features_description)

        nb_tokens_to_mask = tf.cast(tf.reduce_sum(x["masked_lm_weights"]), tf.int32)
        if nb_tokens_to_mask > 1:
            nb_maskable_indices = tf.cast(tf.reduce_sum(x["input_mask"]), tf.int64)
            maskable_indices = tf.range(0, nb_maskable_indices)
            maskable_indices = tf.random.shuffle(maskable_indices)
            masked_lm_positions = maskable_indices[:nb_tokens_to_mask]
        else:
            masked_lm_positions = tf.slice(x["masked_lm_positions"], [0], [nb_tokens_to_mask])

        input_ids = tf.tensor_scatter_nd_update(
            tensor=tf.strings.as_string(x["input_ids"]),
            indices=tf.reshape(masked_lm_positions, (-1, 1)),
            updates=tf.repeat(["[MASK]"], nb_tokens_to_mask)
        )

        if training and nb_tokens_to_mask < 20:
            padding_idx = tf.constant([[0]])
            paddings = tf.reshape(20 - nb_tokens_to_mask, [-1, 1])
            paddings = tf.concat([padding_idx, paddings], axis=1)
            masked_lm_positions = tf.pad(masked_lm_positions, paddings)
        masked_lm_ids = tf.gather(x["input_ids"], masked_lm_positions)

        x["input_ids"] = movie_id_lookup(input_ids)
        x["masked_lm_ids"] = movie_id_lookup(tf.strings.as_string(masked_lm_ids))
        return x

    return _parse_sample
