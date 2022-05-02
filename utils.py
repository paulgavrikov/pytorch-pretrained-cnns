from git import RemoteProgress
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm


class MyCheckpoint(ModelCheckpoint):

    def __init__(self, **kwargs):
        super(MyCheckpoint, self).__init__(**kwargs)

    def on_pretrain_routine_start(self, trainer, pl_module):
        super(MyCheckpoint, self).on_pretrain_routine_start(trainer, pl_module)
        if self.save_top_k == -1:
            monitor_candidates = self._monitor_candidates(trainer)
            last_filepath = self._get_metric_interpolated_filepath_name(
                monitor_candidates, trainer.current_epoch, trainer.global_step, trainer
            )
            self._save_model(last_filepath, trainer, pl_module)


class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        pbar = tqdm(total=max_count)
        pbar.update(cur_count)
