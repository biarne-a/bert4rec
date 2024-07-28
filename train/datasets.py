import re
from typing import Dict

import tensorflow as tf

from config.config import Config


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


def get_data(config: Config):
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    train_ds = _get_dataset_from_files(config, "train")
    val_ds = _get_dataset_from_files(config, "val")
    test_ds = _get_dataset_from_files(config, "test")

    movie_id_vocab = list(unique_train_movie_id_counts.keys()) + ["[MASK]"]
    movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
    reverse_movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab, invert=True)

    def _get_parse_function(features_description):
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, features_description)
        return _parse_function

    train_features_description = config.model_config.get_train_features_description()
    val_and_test_features_description = config.model_config.get_val_and_test_features_description()
    train_parse_function = _get_parse_function(train_features_description)
    val_and_test_parse_function = _get_parse_function(val_and_test_features_description)

    setup_batch_fn = config.model_config.get_setup_batch_fn(config.batch_size, movie_id_lookup)
    train_ds = train_ds.map(train_parse_function).batch(config.batch_size).map(setup_batch_fn).repeat()
    val_ds = (
        val_ds.map(val_and_test_parse_function) #.filter(lambda x: tf.reduce_sum(x["input_mask"]) > 0)
              .batch(config.batch_size).map(setup_batch_fn).repeat()
    )
    test_ds = (
        test_ds.map(val_and_test_parse_function) #.filter(lambda x: tf.reduce_sum(x["input_mask"]) > 0)
               .batch(config.batch_size).map(setup_batch_fn).repeat()
    )

    nb_train = 2_665_787
    nb_test = 162_407
    nb_val = 162_407

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
