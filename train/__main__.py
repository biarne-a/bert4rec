import argparse
import sys

from train.config import Config
from bert4rec.bert_config import BertConfig
from train.run import run_training


def _fetch_class(class_name):
    try:
        return globals()[class_name]
    except KeyError:
        raise ValueError(f"Class '{class_name}' not found")


def _parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, type=str, default="data")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--model_config_cls", required=True, type=str)
    cmd_line_args = vars(parser.parse_args(args=sys.argv[1:]))
    model_config_cls = _fetch_class(cmd_line_args.pop("model_config_cls"))
    return Config(model_config_cls, **cmd_line_args)


def run():
    config = _parse_config()
    run_training(config)


if __name__ == "__main__":
    run()
