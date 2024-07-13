import re
from typing import Dict, Optional

import tensorflow as tf

from bert4rec.train.config import Config


class Data:
    def __init__(
        self,
        train_ds: tf.data.Dataset,
        nb_train: int,
        val_ds: tf.data.Dataset,
        nb_val: int,
        test_ds: tf.data.Dataset,
        nb_test: int,
        movie_id_counts: Dict[str, int],
        movie_id_lookup: tf.keras.layers.StringLookup,
        reverse_movie_id_lookup: tf.keras.layers.StringLookup,
    ):
        self.train_ds = train_ds
        self.nb_train = nb_train
        self.val_ds = val_ds
        self.nb_val = nb_val
        self.test_ds = test_ds
        self.nb_test = nb_test
        self.movie_id_counts = movie_id_counts
        self.movie_id_lookup = movie_id_lookup
        self.reverse_movie_id_lookup = reverse_movie_id_lookup

    @property
    def vocab_size(self):
        return self.movie_id_lookup.vocab_size()


def _read_unique_train_movie_id_counts(bucket_dir):
    with tf.io.gfile.GFile(f"{bucket_dir}/vocab/train_movie_counts.txt-00000-of-00001") as f:
        unique_train_movie_id_counts = {}
        for line in f.readlines():
            match = re.match("^\(([0-9]+), ([0-9]+)\)$", line.strip())  # noqa: W605
            movie_id = match.groups()[0]
            count = int(match.groups()[1])
            unique_train_movie_id_counts[movie_id] = count
    return unique_train_movie_id_counts


def _get_dataset_from_files(config: Config, dataset_type: str):
    filenames = f"{config.data_dir}/tfrecords/{dataset_type}/*.gz"
    dataset = tf.data.Dataset.list_files(filenames, seed=Config.SEED)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset


def _get_features_description(config: Config, nb_max_masked_ids_per_seq: Optional[int] = None) -> Dict[str, tf.io.FixedLenFeature]:
    max_seq_length = config.bert_config.max_sequence_length
    nb_max_masked_ids_per_seq = nb_max_masked_ids_per_seq or config.nb_max_masked_ids_per_seq
    return {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([nb_max_masked_ids_per_seq], tf.float32),
    }


def keep_non_zero_values(tensor):
    # Reverse the tensor
    reversed_tensor = tf.reverse(tensor, axis=[0])

    # Find the first non-zero element's index
    first_non_zero_index = tf.argmax(tf.cast(reversed_tensor != 0, tf.int32))

    # Compute the length to keep (number of elements from the start to the first non-zero element)
    length_to_keep = tf.shape(tensor)[0] - tf.cast(first_non_zero_index, tf.int32)

    # Slice the original tensor to keep only the non-zero elements
    truncated_tensor = tf.slice(tensor, [0], [length_to_keep])

    return length_to_keep, truncated_tensor


def get_data(config: Config):
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    train_ds = _get_dataset_from_files(config, "train")
    val_ds = _get_dataset_from_files(config, "val")
    test_ds = _get_dataset_from_files(config, "test")

    train_features_description = _get_features_description(config)
    val_and_test_features_description = _get_features_description(config, nb_max_masked_ids_per_seq=1)

    movie_id_vocab = list(unique_train_movie_id_counts.keys()) + ["[MASK]"]
    movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
    reverse_movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab, invert=True)

    def _get_parse_function(features_description):
        def _parse_function(example_proto):
            x = tf.io.parse_single_example(example_proto, features_description)

            # Replace masked input ids with the [MASK] token. Without that, learning is trivial as you give the answer
            # in the input
            masked_length, masked_lm_positions = keep_non_zero_values(x["masked_lm_positions"])
            input_ids = tf.tensor_scatter_nd_update(
                tensor=tf.strings.as_string(x["input_ids"]),
                indices=tf.reshape(masked_lm_positions, (-1, 1)),
                updates=tf.repeat(["[MASK]"], masked_length)
            )
            x["input_ids"] = movie_id_lookup(input_ids)
            x["masked_lm_ids"] = movie_id_lookup(tf.strings.as_string(x["masked_lm_ids"]))
            return x
        return _parse_function

    train_parse_function = _get_parse_function(train_features_description)
    val_and_test_parse_function = _get_parse_function(val_and_test_features_description)

    train_ds = train_ds.map(train_parse_function).repeat().batch(config.batch_size)
    val_ds = val_ds.map(val_and_test_parse_function).repeat().batch(config.batch_size)
    test_ds = test_ds.map(val_and_test_parse_function).repeat().batch(config.batch_size)

    nb_train = 42_279_870
    nb_test = 162_442
    nb_val = 162_473

    return Data(
        train_ds,
        nb_train,
        val_ds,
        nb_val,
        test_ds,
        nb_test,
        unique_train_movie_id_counts,
        movie_id_lookup,
        reverse_movie_id_lookup
    )
