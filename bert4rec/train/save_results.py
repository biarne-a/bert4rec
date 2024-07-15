import os
import pickle

import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm

from bert4rec.train.config import Config
from bert4rec.train.datasets import Data
from bert4rec.train.bert4rec_model import BERT4RecModel



def save_history(history: keras.callbacks.History, config: Config):
    results_dir = f"{config.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = f"{results_dir}/history_bert4rec_training.p"
    pickle.dump(history.history, tf.io.gfile.GFile(output_file, "wb"))


def save_predictions(config: Config, data: Data, model: BERT4RecModel, k: int = 10):
    nb_test_batches = data.nb_test // config.batch_size
    local_filename = f"{config.data_dir}/results/predictions_bert4rec.csv"
    with tf.io.gfile.GFile(local_filename, "w") as fileh:
        columns = ["label"] + [f"output_{i}" for i in range(k)]
        header = ",".join(columns)
        fileh.write(f"{header}\n")
        i_batch = 0
        for batch in tqdm(data.test_ds.as_numpy_iterator(), total=nb_test_batches):
            logits = model.predict_on_batch(batch)
            top_indices = tf.math.top_k(logits, k=k).indices
            top_predictions = data.reverse_movie_id_lookup(top_indices).numpy().reshape((-1, k))
            y_true = batch["masked_lm_ids"]
            y_true = data.reverse_movie_id_lookup(y_true)
            predictions_numpy = np.concatenate((y_true, top_predictions), axis=1)
            np.savetxt(fileh, predictions_numpy.astype(str), fmt="%s", delimiter=",")
            i_batch += 1
            if i_batch == nb_test_batches:
                break
            fileh.flush()
    return local_filename
