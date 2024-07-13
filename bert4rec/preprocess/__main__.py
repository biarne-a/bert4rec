import random
# from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
import apache_beam as beam
from apache_beam.pvalue import PCollection


class SampleType: #(Enum):
    TRAIN_SLIDING = 0
    TRAIN_LAST = 1
    VALIDATION = 2
    TEST = 3


def _sort_views_by_timestamp(group) -> List[int]:
    views = group[0][1]
    views.sort(key=lambda x: x[-1])
    return [v[0] for v in views]


def _generate_examples_from_complete_sequences(
    complete_sequence: List[int],
    max_context_len: int,
    proporition_sliding_window: float,
):
    """
    Generate user sequences from a single complete user sequence using a sliding window.
    If user complete sequence is shorter than max_context_len, sequence will be padded with 0s.

    :param complete_sequence: The complete sequence to generate examples from (A list of ids).
    :param max_context_len: The maximum length of the context.
    :param sliding_window_step_size: The size of the step in the sliding window.
    :param duplication_factor: The number of time each sequence should be duplicated (considering different random input
    will be masked)
    :param nb_max_masked_ids_per_seq: The maximal number of ids that can be masked for prediction in a sequence
    :param masked_lm_prob: The probability that an id in a sequence will be masked for prediction

    :return: examples: Generated examples from this single timeline.
    """
    examples = []
    sliding_window_step_size = max_context_len
    if proporition_sliding_window != -1.0:
        sliding_window_step_size = int(proporition_sliding_window * max_context_len)

    # The last 2 tokens of each sequence is for validation and testing
    train_sequence_len = len(complete_sequence) - 2
    train_complete_sequence = complete_sequence[:-2]

    start_indexes = list(range(train_sequence_len - max_context_len, 0, -sliding_window_step_size))
    start_indexes.append(0)
    for start_idx in start_indexes[::-1]:
        example = _get_new_example(train_complete_sequence, start_idx, max_context_len, sample_type=0)
        examples.append(example)

    # Add a train sequence for last index masking
    example = _get_new_example(train_complete_sequence, start_idx, max_context_len, sample_type=1)
    examples.append(example)

    # Add validation sequence
    start_idx_validation = len(complete_sequence) - max_context_len - 1
    example = _get_new_example(complete_sequence, start_idx_validation, max_context_len, sample_type=2)
    examples.append(example)

    # Add test sequence
    start_idx_test = len(complete_sequence) - max_context_len
    example = _get_new_example(complete_sequence, start_idx_test, max_context_len, sample_type=3)
    examples.append(example)

    return examples


def _get_new_example(complete_sequence, start_idx, max_context_len, sample_type):
    end_idx = start_idx + max_context_len
    input_ids = complete_sequence[start_idx:end_idx]
    input_mask = [1] * len(input_ids)
    # Pad sequence with 0s.
    while len(input_ids) < max_context_len:
        input_ids.append(0)
        input_mask.append(0)
    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "sample_type": sample_type,
    }


def _augment_training_sequences_and_set_masks(
    sample: Dict[str, Any],
    duplication_factor: int,
    nb_max_masked_ids_per_seq: int,
    mask_ratio: float,
):
    """

    """
    import random

    all_augmented_samples = []
    if sample["sample_type"] == 0:
        input_ids = list(sample["input_ids"])
        nb_filled_input_ids = sum(sample["input_mask"])
        for _ in range(duplication_factor):
            nb_ids_to_mask = min(nb_max_masked_ids_per_seq, max(1, int(nb_filled_input_ids * mask_ratio)))

            masked_lm_ids = []
            masked_lm_positions = []
            masked_lm_weights = []

            # Shuffle the positions
            shuffled_id_positions = list(range(nb_filled_input_ids))
            random.shuffle(shuffled_id_positions)

            # And take the required number of masked ids
            for idx in range(nb_ids_to_mask):
                masked_position = shuffled_id_positions[idx]
                masked_lm_id = input_ids[idx]
                masked_lm_positions.append(masked_position)
                masked_lm_ids.append(masked_lm_id)
                masked_lm_weights.append(1.0)

            # Pad the masks to obtain a complete sequence up to the maximum allowed
            while len(masked_lm_positions) < nb_max_masked_ids_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            augmented_sample = {
                "input_ids": input_ids,
                "input_mask": sample["input_mask"],
                "sample_type": sample["sample_type"],
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights,
            }
            all_augmented_samples.append(augmented_sample)
    else:
        # Mask last position to match the val and test sets settings
        last_position_masked_sample = _set_mask_last_position(sample)
        all_augmented_samples.append(last_position_masked_sample)
    return all_augmented_samples


def _set_mask_last_position(sample: Dict[str, Any]):
    input_ids = sample["input_ids"]
    nb_filled_input_ids = sum(sample["input_mask"])
    masked_lm_position = nb_filled_input_ids - 1
    return {
        "input_ids": input_ids,
        "input_mask": sample["input_mask"],
        "sample_type": sample["sample_type"],
        "masked_lm_positions":  [masked_lm_position],
        "masked_lm_ids": [input_ids[masked_lm_position]],
        "masked_lm_weights": [1.0]
    }


def _count_movies_in_ratings(train_samples: PCollection):
    return (
            train_samples
            | "Flatten train samples for count" >> beam.FlatMap(lambda x: x["input_ids"])
            | "Set Movie Id Key" >> beam.Map(lambda x: (x, 1))
            | "Count By Movie Id" >> beam.combiners.Count.PerKey()
            | "Remove 0 counts" >> beam.Filter(lambda x: x[0] != 0)
    )


def _serialize_in_tfrecords(x):
    import tensorflow as tf

    feature = {
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_ids"])),
        "input_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=x["input_mask"])),
        "masked_lm_positions": tf.train.Feature(int64_list=tf.train.Int64List(value=x["masked_lm_positions"])),
        "masked_lm_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=x["masked_lm_ids"])),
        "masked_lm_weights": tf.train.Feature(float_list=tf.train.FloatList(value=x["masked_lm_weights"])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def _save_in_tfrecords(data_dir: str, examples: PCollection, data_desc: str):
    output_dir = f"{data_dir}/tfrecords/{data_desc}"
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    prefix = f"{output_dir}/data"
    examples | f"Write {data_desc} examples" >> beam.io.tfrecordio.WriteToTFRecord(
        prefix,
        file_name_suffix=".tfrecord.gz",
    )


def _save_train_movie_counts(data_dir: str, counts: PCollection):
    counts | "Write train movie counts" >> beam.io.WriteToText(f"{data_dir}/vocab/train_movie_counts.txt", num_shards=1)


def _transform_to_rating(csv_row):
    cells = csv_row.split(",")
    return {"userId": int(cells[0]), "movieId": int(cells[1]), "rating": float(cells[2]), "timestamp": int(cells[3])}


def _filter_examples_per_sample_type(examples_per_user: PCollection, sample_type: int) -> PCollection:
    return examples_per_user | f"Filter {sample_type}" >> beam.Filter(lambda x: x["sample_type"] == sample_type)


def _filter_examples_per_sample_types(examples_per_user: PCollection, sample_type1: int, sample_type2: int) -> PCollection:
    return examples_per_user | f"Filter {sample_type1} and {sample_type2}" >> beam.Filter(lambda x: (x["sample_type"] == sample_type1) or (x["sample_type"] == sample_type2))


def preprocess_with_dataflow(
    data_dir: str,
    max_context_len: int,
    proporition_sliding_window: float,
    duplication_factor: int,
    nb_max_masked_ids_per_seq: int,
    mask_ratio: float,
    implicit_rating_threshold: float,
):
    """
    Preprocess the data: read ratings from CSV file and transform them into ready to train serialized tensors in
    Tensorflow tensor records format.
    :param data_dir: The directory from where to find the ratings CSV file
    :param max_context_len: The maximum length a user sequence can have
    :param sliding_window_step_size: The size of the step in the sliding window used to create the training sequences
    from a complete user sequence.
    :param duplication_factor: The number of time each sequence should be duplicated (considering different random input
    will be masked)
    :param nb_max_masked_ids_per_seq: The maximal number of ids that can be masked in a sequence
    :param mask_ratio: The ratio of input ids to mask for prediction
    :param implicit_rating_threshold: The threshold used to decide whether a rating is an implicit positive sample or
    negative
    """
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/adrienbiarnes/Projects/bert4rec/credentials.json"

    options = beam.pipeline.PipelineOptions(
        runner="DataflowRunner",
        experiments=["use_runner_v2"],
        project="concise-haven-277809",
        staging_location="gs://movie-lens-25m/beam/stg",
        temp_location="gs://movie-lens-25m/beam/tmp",
        job_name="ml-25m-preprocess",
        num_workers=8,
        region="us-central1",
        sdk_container_image="northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/bert4rec-preprocess",
    )
    with beam.Pipeline(options=options) as pipeline:
        raw_ratings = (
            pipeline
            | "Read ratings CSV" >> beam.io.textio.ReadFromText(f"{data_dir}/ratings.csv", skip_header_lines=1)
            | "Transform row to rating dict" >> beam.Map(_transform_to_rating)
            | "Filter low ratings (keep implicit positives)" >> beam.Filter(lambda x: x["rating"] > implicit_rating_threshold)
        )

        user_complete_sequences = (
            raw_ratings
            | "Select columns" >> beam.Map(lambda x: (x["userId"], (x["movieId"], x["timestamp"])))
            | "Group By User Id" >> beam.GroupByKey()
            | "Add Views Counts" >> beam.Map(lambda x: (x, len(x[1])))
            | "Filter If Not Enough Views" >> beam.Filter(lambda x: x[1] >= 5)
            | "Sort Views By Timestamp" >> beam.Map(_sort_views_by_timestamp)
        )

        examples = (
            user_complete_sequences
            | "Generate examples from complete sequences" >>
            beam.Map(
                _generate_examples_from_complete_sequences,
                max_context_len=max_context_len,
                proporition_sliding_window=proporition_sliding_window,
            )
            | f"Flatten examples" >> beam.FlatMap(lambda x: x)
        )

        # Split examples
        train_examples = _filter_examples_per_sample_types(examples, SampleType.TRAIN_SLIDING, SampleType.TRAIN_LAST)
        val_examples = _filter_examples_per_sample_type(examples, SampleType.VALIDATION)
        test_examples = _filter_examples_per_sample_type(examples, SampleType.TEST)

        # Add masks and augment training data
        train_examples = (
            train_examples
            | "Augment training data and set masks" >> beam.Map(
                _augment_training_sequences_and_set_masks,
                duplication_factor=duplication_factor,
                nb_max_masked_ids_per_seq=nb_max_masked_ids_per_seq,
                mask_ratio=mask_ratio,
            )
            | "Flatten augmented training examples" >> beam.FlatMap(lambda x: x)
        )
        val_examples = val_examples | "Set mask last position - val" >> beam.Map(_set_mask_last_position)
        test_examples = test_examples | "Set mask last position - test" >> beam.Map(_set_mask_last_position)

        # Serialize
        train_tf_examples = train_examples | "Serialize train" >> beam.Map(_serialize_in_tfrecords) | beam.Reshuffle()
        val_tf_examples = val_examples | "Serialize val" >> beam.Map(_serialize_in_tfrecords)
        test_tf_examples = test_examples | "Serialize test" >> beam.Map(_serialize_in_tfrecords)

        # Save to disk
        _save_in_tfrecords(data_dir, train_tf_examples, data_desc="train")
        _save_in_tfrecords(data_dir, val_tf_examples, data_desc="val")
        _save_in_tfrecords(data_dir, test_tf_examples, data_desc="test")

        # Count vocab
        train_movie_counts = _count_movies_in_ratings(train_examples)
        _save_train_movie_counts(data_dir, train_movie_counts)


if __name__ == "__main__":
    preprocess_with_dataflow(
        data_dir="gs://movie-lens-25m",
        max_context_len=200,
        proporition_sliding_window=0.5,
        duplication_factor=10,
        nb_max_masked_ids_per_seq=20,
        mask_ratio=0.2,
        implicit_rating_threshold=2.0,
    )
