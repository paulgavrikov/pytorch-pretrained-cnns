from git import RemoteProgress
import torch
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, base
import argparse
import datetime


class ExtendedModelCheckpoint(ModelCheckpoint):
    
    CHECKPOINT_NAME_FIRST = "first"
    
    def __init__(self, save_first=False, **kwargs):
        super().__init__(**kwargs)
        self.save_first = save_first
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        if self.save_first:
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_FIRST)
            self._save_checkpoint(trainer, filepath)


class TimeMonitor(base.Callback):

    def __init__(self):
        super(TimeMonitor, self).__init__()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log("start_time", datetime.datetime.now().timestamp())

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log("end_time", datetime.datetime.now().timestamp())


class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        pbar = tqdm(total=max_count)
        pbar.update(cur_count)

    
class NormalizedModel(torch.nn.Module):
    
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(-1, 1, 1), requires_grad=False)
        
    def forward(self, x):
        out = (x - self.mean) / self.std 
        out = self.model(out)
        return out
    
    
def none_or_str(value):  # from https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    if value == 'None':
        return None
    return value


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
