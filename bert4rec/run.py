import sys
import os
import random

import numpy as np
import tensorflow as tf

from bert4rec.config import Config
from bert4rec.metrics import MaskedRecall, MaskedMeanAveragePrecision
from bert4rec.datasets import Data, get_data
from bert4rec.bert4rec_model import BERT4RecModel
from bert4rec.save_results import save_history, save_predictions


def build_model(data: Data, config: Config):
    bert_config = config.bert_config.to_dict()
    return BERT4RecModel(data.vocab_size, **bert_config)


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
    return f"{config.data_dir}/results/bert4rec.model"


def run_training(config: Config):
    data = get_data(config)
    model = build_model(data, config)
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
        epochs=1_000,
        steps_per_epoch=5_000,
        validation_data=data.val_ds,
        validation_steps=data.nb_val // config.batch_size,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir="logs", update_freq=100),
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(save_filepath, save_best_only=True),
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
    save_predictions(config, data, model)
