import argparse
import sys

from bert4rec.config import Config
from bert4rec.run import run_training


def _parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, type=str, default="data")
    parser.add_argument("--bert_config_file", required=True, type=str)
    cmd_line_args = vars(parser.parse_args(args=sys.argv[1:]))
    return Config(**cmd_line_args)


def run():
    config = _parse_config()
    run_training(config)


if __name__ == "__main__":
    run()
