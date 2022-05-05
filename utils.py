from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar

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

            
class MyProgressBar(RichProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False
        
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items["val_acc_max"] = pl_module.acc_max
        return items