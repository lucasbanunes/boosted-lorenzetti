from lightning.pytorch.callbacks import TQDMProgressBar


class LowVerboseProgressBar(TQDMProgressBar):
    """
    Not even a little bit less verbose. Have to figure out something afterwards
    """

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
