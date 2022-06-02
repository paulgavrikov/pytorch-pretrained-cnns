import os
from argparse import ArgumentParser
import json
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from module import TrainModule
import data as datasets
import models
from utils import *
from pprint import pprint
import wandb


def start_training(args):
    seed_everything(args["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]

    data_dir = os.path.join(args["data_dir"], args["dataset"])
    data = datasets.get_dataset(args["dataset"])(data_dir, args["batch_size"], args["num_workers"])

    args["num_classes"] = data.num_classes
    args["in_channels"] = data.in_channels

    model = TrainModule(args)
    if args["load_checkpoint"] is not None:
        state = torch.load(args["load_checkpoint"], map_location=model.device)

        if args["replace_fc"]:
            num_classes = state['hyper_parameters']['hparams']['num_classes']
            in_channels = state['hyper_parameters']['hparams']['in_channels']

            args["num_classes"] = num_classes
            args["in_channels"] = in_channels

            model = TrainModule(args)

        if "state_dict" in state:
            state = state["state_dict"]

        model.model.load_state_dict(
            dict((key.replace("model.", "").replace("classifier", "fc"), value) for (key, value) in
                 state.items()))

        if args["reset_head"]:
            model.model.fc.reset_parameters()

        if args["replace_fc"]:
            # Replace only the last element from the sequential fc
            if isinstance(model.model.fc, torch.nn.Sequential):
                model.model.fc[-1] = torch.nn.Linear(model.model.fc[-1].in_features, data.num_classes)
            else:
                model.model.fc = torch.nn.Linear(model.model.fc.in_features, data.num_classes)

    loggers = []
    csv_logger = CSVLogger(os.path.join(args["output_dir"], args["dataset"]), args["classifier"] + args["postfix"])
    csv_logger.save()
    loggers.append(csv_logger)

    if args["wandb"]:
        wandb_logger = WandbLogger(project=args["wandb"], log_model=False)
        wandb.run.name = f"{args['classifier']}-{args['dataset']}-{wandb.run.id}"
        wandb.run.save()
        loggers.append(wandb_logger)

    callbacks = []

    if args["checkpoints"]:
        checkpoint_cb = ExtendedModelCheckpoint(save_first=True, monitor="acc/val", mode="max", save_top_k=1,
                                                save_last=args["checkpoints"] == "last_best")
        callbacks.append(checkpoint_cb)

    progress_bar_cb = RichProgressBar()
    callbacks.append(progress_bar_cb)

    trainer = Trainer(
        fast_dev_run=False,
        logger=loggers,
        gpus=-1,
        deterministic=not args["cudnn_non_deterministic"],
        benchmark=True,
        enable_model_summary=False,
        log_every_n_steps=10,
        max_epochs=args["max_epochs"],
        enable_checkpointing=args["checkpoints"] is not None,
        precision=args["precision"],
        callbacks=callbacks,
        profiler=args["profiler"],
        num_sanity_val_steps=0  # sanity check must be turned off or bad performance callback will trigger.
    )
    if args["verbose"]:
        print()
        print("ARGS:")
        pprint(args)
        print()
        print("MODEL:")
        pprint(model.model)
    trainer.fit(model, data)


def dump_info():
    print("Available models:")
    for x in models.all_classifiers.keys():
        print(f"\t{x}")
    print()
    print("Available data sets:")
    for x in datasets.all_datasets.keys():
        print(f"\t{x}")


def prepare_data(args):
    data_dir = os.path.join(args["data_dir"], args["dataset"])
    data = datasets.get_dataset(args["dataset"])(data_dir, 1, 0)
    next(iter(data.train_dataloader()))
    next(iter(data.val_dataloader()))
    print("Dataset is ready.")


def main(args):
    if type(args) is not dict:
        args = vars(args)

    if args["mode"] == "train":
        start_training(args)
    elif args["mode"] == "initdata":
        prepare_data(args)
    elif args["mode"] == "info":
        dump_info()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "info", "initdata"])

    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--params", type=str, default=None)  # load params from json

    parser.add_argument("--checkpoints", type=str, default="last_best", choices=["all", "last_best", None])

    parser.add_argument("--classifier", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--reset_head", type=str2bool, default=False)
    parser.add_argument("--replace_fc", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--postfix", type=str, default="")

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--cudnn_non_deterministic", type=str2bool, default=True)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adagrad", "rmsprop", "adamw"])
    parser.add_argument("--scheduler", type=none_or_str, default=None,
                        choices=["WarmupCosine", "Step", "FrankleStep", "None", None])
    parser.add_argument("--freeze", type=none_or_str, default=None, choices=["conv", "None", None])
    parser.add_argument("--cutmix_prob", type=float, default=0)
    parser.add_argument("--aux_loss", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--profiler", type=str, default=None)
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--wandb_sweepid", type=str, default=None)

    parser.add_argument("--extra1", type=str, default=None)
    parser.add_argument("--extra2", type=str, default=None)

    _args = parser.parse_args()

    if _args.params is not None:
        json_args = argparse.Namespace()
        with open(_args.params, "r") as f:
            json_args.__dict__ = json.load(f)

        _args = parser.parse_args(namespace=json_args)

    main(_args)
