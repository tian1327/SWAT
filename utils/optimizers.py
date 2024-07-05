from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

"""
Borrowed from: https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/engine/optimizer/scheduler.py
"""

class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def get_optimizer(params, optim_type, wd,):
    if optim_type == 'SGD':        
        # return optim.SGD(params, lr=lr, momentum = 0.9, weight_decay=wd)
        for param in params:
            param['momentum'] = 0.9
            param['weight_decay'] = wd
        return optim.SGD(params)        
        
    elif optim_type == 'AdamW':
        # return optim.AdamW(params, lr=lr, betas=(0.9,0.999), weight_decay=wd)
        # return optim.AdamW(params, betas=(0.9,0.999), weight_decay=wd)

        for param in params:
            param['betas'] = (0.9,0.999)
            param['weight_decay'] = wd
        return optim.AdamW(params)

def get_warmup_scheduler(optimizer, scheduler, warmup_iter = 50, warmup_lr = 1e-6):
    return LinearWarmupScheduler(
        optimizer=optimizer,
        successor=scheduler,
        warmup_epoch=warmup_iter,
        min_lr=warmup_lr
    )