import argparse
import sys

from bert4rec.train.config import Config
from bert4rec.train.run import run_training


def _parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, type=str, default="data")
    parser.add_argument("--bert_config_file", required=True, type=str)
    parser.add_argument("--batch_size", required=False, type=int, default=256)
    parser.add_argument("--nb_max_masked_ids_per_seq", required=False, type=int, default=20)
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-4)
    cmd_line_args = vars(parser.parse_args(args=sys.argv[1:]))
    return Config(**cmd_line_args)


def run():
    config = _parse_config()
    run_training(config)


if __name__ == "__main__":
    run()
