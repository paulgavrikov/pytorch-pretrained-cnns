from git import RemoteProgress
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm


class MyCheckpoint(ModelCheckpoint):

    def __init__(self, **kwargs):
        super(MyCheckpoint, self).__init__(**kwargs)

    def on_fit_start(self, trainer, pl_module):
        super(MyCheckpoint, self).on_pretrain_routine_start(trainer, pl_module)
        if self.save_top_k == -1:
            monitor_candidates = self._monitor_candidates(trainer)
            last_filepath = self._get_metric_interpolated_filepath_name(
                monitor_candidates, trainer
            )
            self._save_checkpoint(trainer, last_filepath)


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
