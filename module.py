import pytorch_lightning as pl
import torch
import torchmetrics
from scheduler import WarmupCosineLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import models
from pprint import pprint
import torch.nn.functional as F


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class TrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.myhparams = hparams
        self.save_hyperparameters()
        self.model = models.get_model(self.myhparams["classifier"])(in_channels=hparams["in_channels"],
                                                                    num_classes=hparams["num_classes"])
        self.acc_max = 0
        self.cutmix_beta = 1

    def forward(self, batch):
        images, labels = batch
        if self.myhparams["aux_loss"] == 0 or not self.model.training:
            r = np.random.rand(1)
            if self.cutmix_beta > 0 and r < self.myhparams["cutmix_prob"]:
                lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).to(images.device)
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(ims.size(), lam)
                ims[:, :, bbx1:bbx2, bby1:bby2] = ims[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                predictions = self.model(images)
                loss = F.cross_entropy(predictions, target_a) * lam + F.cross_entropy(predictions, target_b) * (
                        1. - lam)
            else:
                predictions = self.model(images)
                loss = F.cross_entropy(predictions, labels)
        else:
            predictions, aux_outputs = self.model(images)
            loss = F.cross_entropy(predictions, labels) ** 2
            for aux_output in aux_outputs:
                loss += F.cross_entropy(aux_output, labels) ** 2
            loss = torch.sqrt(loss)

        accuracy = torchmetrics.functional.accuracy(predictions, labels)
        return {"loss": loss, "accuracy": accuracy * 100}

    def training_step(self, batch, batch_nb):
        return self.forward(batch)

    def training_epoch_end(self, outs):
        loss = torch.stack([x["loss"] for x in outs]).mean().item()
        accuracy = torch.stack([x["accuracy"] for x in outs]).mean().item()

        self.log("loss/train", loss)
        self.log("acc/train", accuracy, prog_bar=True)

    def validation_step(self, batch, batch_nb):
        return self.forward(batch)

    def training_step_end(self, outs):
        self.log("acc/train", outs["accuracy"], prog_bar=True)
        self.log("loss/train", outs["loss"])

    def validation_epoch_end(self, outs):
        loss = torch.stack([x["loss"] for x in outs]).mean().item()
        accuracy = torch.stack([x["accuracy"] for x in outs]).mean().item()

        self.log("loss/val", loss)
        if accuracy > self.acc_max:
            self.acc_max = accuracy

        self.log("acc_max/val", self.acc_max, prog_bar=True)
        self.log("acc/val", accuracy, prog_bar=True)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):

        if self.myhparams["freeze"] == "conv":
            for module in self.model.modules():
                if type(module) == torch.nn.Conv2d:
                    for param in module.parameters():
                        param.requires_grad = False

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.myhparams["verbose"]:
            print()
            print("TRAINABLE PARAMETERS:")
            pprint([f"{name} {p.shape}" for name, p in
                    filter(lambda p: p[1].requires_grad, self.model.named_parameters())])
            print(
                f"TOTAL: {sum(list(map(lambda p: p.numel(), filter(lambda p: p.requires_grad, self.model.parameters()))))}")

        optimizers, schedulers = [], []

        if self.myhparams["optimizer"] == "sgd":
            optimizers.append(torch.optim.SGD(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"],
                momentum=self.myhparams["momentum"],
                nesterov=True
            ))
        elif self.myhparams["optimizer"] == "adam":
            optimizers.append(torch.optim.Adam(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"]
            ))
        elif self.myhparams["optimizer"] == "adagrad":
            optimizers.append(torch.optim.Adagrad(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"]
            ))
        elif self.myhparams["optimizer"] == "rmsprop":
            optimizers.append(torch.optim.RMSprop(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"]
            ))
        elif self.myhparams["optimizer"] == "adamw":
            optimizers.append(torch.optim.AdamW(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"]
            ))

        if self.myhparams["scheduler"] == "WarmupCosine":
            total_steps = self.myhparams["max_epochs"] * len(self.train_dataloader())
            schedulers.append({
                "scheduler": WarmupCosineLR(optimizers[0], warmup_epochs=total_steps * 0.3, max_epochs=total_steps),
                "interval": "step",
                "name": "learning_rate",
            })
        elif self.myhparams["scheduler"] == "Step":
            schedulers.append({
                "scheduler": StepLR(optimizers[0], step_size=30, gamma=0.1),
                "interval": "epoch",
                "name": "learning_rate",
            })
        elif self.myhparams["scheduler"] == "FrankleStep":
            schedulers.append({
                "scheduler": MultiStepLR(optimizers[0], milestones=[80, 120], gamma=0.1),
                "interval": "epoch",
                "name": "learning_rate",
            })

        return optimizers, schedulers
