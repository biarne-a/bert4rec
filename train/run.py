import sys
import os
import random

import numpy as np
import tensorflow as tf

from config.config import Config
from train.metrics import MaskedRecall, MaskedMeanAveragePrecision
from train.datasets import get_data
from train.save_results import save_history, save_predictions


def _debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def set_seed():
    # for tf.random
    tf.random.set_seed(Config.SEED)
    # for numpy.random
    np.random.seed(Config.SEED)
    # for built-in random
    random.seed(Config.SEED)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(Config.SEED)


def _get_model_save_filepath(config: Config) -> str:
    return f"{config.results_dir}/model.keras"


def run_training(config: Config):
    os.makedirs(config.results_dir, exist_ok=True)

    data = get_data(config)
    model = config.model_config.build_model(data)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[
            MaskedRecall(k=10),
            MaskedMeanAveragePrecision(k=10)
        ],
        run_eagerly=_debugger_is_active(),
    )
    save_filepath = _get_model_save_filepath(config)
    history = model.fit(
        x=data.train_ds,
        epochs=1,#1_000,
        steps_per_epoch=3,#data.nb_train_batches,
        validation_data=data.val_ds,
        validation_steps=3,#data.nb_val_batches,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir="logs", update_freq=100),
            tf.keras.callbacks.EarlyStopping(monitor="val_recall_at_10", mode="max", patience=2),
            tf.keras.callbacks.ModelCheckpoint(save_filepath, monitor="val_recall_at_10", mode="max", save_best_only=True),
        ],
        verbose=1,
    )
    save_history(history, config)
    run_evaluation(config)


def run_evaluation(config: Config):
    data = get_data(config)
    save_filepath = _get_model_save_filepath(config)
    model = tf.keras.models.load_model(save_filepath, custom_objects={
        "MaskedRecall": MaskedRecall,
        "MaskedMeanAveragePrecision": MaskedMeanAveragePrecision,
    })
    model.evaluate(data.test_ds, steps=data.nb_test_batches)
    save_predictions(config, data, model)
