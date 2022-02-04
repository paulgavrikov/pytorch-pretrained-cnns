import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from scheduler import WarmupCosineLR
import numpy as np
import models


class TrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.myparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.model = models.get_model(self.myparams["classifier"])(in_channels=hparams["in_channels"],
                                                                  num_classes=hparams["num_classes"])
        self.acc_max = 0
        
    def forward(self, batch, metric=None):
        images, labels = batch
        if self.myparams["aux_loss"] == 0 or not self.model.training:
            predictions = self.model(images)
            loss = torch.nn.CrossEntropyLoss()(predictions, labels)
        else:
            predictions, aux_outputs = self.model(images)
            loss = torch.nn.CrossEntropyLoss()(predictions, labels) ** 2
            for aux_output in aux_outputs:
                loss += torch.nn.CrossEntropyLoss()(aux_output, labels) ** 2
            loss = torch.sqrt(loss)
        if metric is not None:
            accuracy = metric(predictions, labels)
            return loss, accuracy * 100
        else:
            return loss

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch, self.train_accuracy)
        return loss

    def training_epoch_end(self, outs):            
        self.log("loss/train", np.mean([d["loss"].item() for d in outs]))
        self.log("acc/train", self.train_accuracy.compute() * 100)
        self.train_accuracy.reset()
    
    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch, self.val_accuracy)
        return loss

    def validation_epoch_end(self, outs):
        self.log("loss/val", np.mean([d.item() for d in outs]))
        
        acc = self.val_accuracy.compute() * 100
        print(acc)
        if acc > self.acc_max:
            self.acc_max = acc
        
        self.log("acc_max/val", self.acc_max)
        self.log("acc/val", acc)        
        self.val_accuracy.reset()

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.myparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.myparams["learning_rate"],
                weight_decay=self.myparams["weight_decay"],
                momentum=self.myparams["momentum"],
                nesterov=True
            )

            total_steps = self.myparams["max_epochs"] * len(self.train_dataloader())
            scheduler = {
                "scheduler": WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps),
                "interval": "step",
                "name": "learning_rate",
            }

            return [optimizer], [scheduler]
        else:
            optimizer = torch.optim.Adam(
                params,
                lr=self.myparams["learning_rate"],
                weight_decay=self.myparams["weight_decay"]
            )

            return [optimizer], []
