import os
from argparse import ArgumentParser
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from module import TrainModule
import data as datasets
from utils import *
import logging
logging.getLogger('lightning').setLevel(0)


def start_training(args):
    data_dir = os.path.join(args["data_dir"], args["dataset"])
    data = datasets.get_dataset(args["dataset"])(data_dir, args["batch_size"], args["num_workers"])

    args["num_classes"] = data.num_classes
    args["in_channels"] = data.in_channels

    model = TrainModule(args)
    if args["load_checkpoint"] is not None:
        state = torch.load(args["load_checkpoint"], map_location=model.device)
        model.model.load_state_dict(dict((key.replace("model.", ""), value) for (key, value) in
                                         state["state_dict"].items()))

    logger = CSVLogger(args["dataset"], args["classifier"])
        
    checkpoint = MyCheckpoint(monitor="acc_max/val", mode="max", save_top_k=-1 if args["checkpoints"] == "all" else 1,
                              period=1)

    trainer = Trainer(
        fast_dev_run=False,
        logger=logger,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args["max_epochs"],
        checkpoint_callback=False if args["checkpoints"] is None else checkpoint,
        precision=args["precision"],
        callbacks=None,
        num_sanity_val_steps=0  # sanity check must be turned off or bad performance callback will trigger.
    )

    trainer.fit(model, data)


def main(args):
    if type(args) is not dict:
        args = vars(args)

    seed_everything(args["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]
    start_training(args)

        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--params", type=str, default=None)  # load params from json

    parser.add_argument("--checkpoints", type=str, default="last_best", choices=["all", "last_best", None])
    parser.add_argument("--classifier", type=str, default="lowres_resnet9")
    parser.add_argument("--dataset", type=str, default="omniglot")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--seed", type=int, default=0)

    _args = parser.parse_args()
    
    if _args.params is not None:
        json_args = argparse.Namespace()
        with open(_args.params, 'r') as f:
            json_args.__dict__ = json.load(f)

        _args = parser.parse_args(namespace=json_args)
    
    main(_args)