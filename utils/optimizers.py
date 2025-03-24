from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import numpy as np

"""
partially borrowed from: https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/engine/optimizer/scheduler.py
"""


def lock_text_tower(model):

    for m in [model.transformer, model.token_embedding, model.positional_embedding, model.ln_final, model.text_projection]:
        if type(m) is nn.Parameter:
            m.requires_grad = False
        else:
            for p in m.parameters():
                p.requires_grad = False


def set_optimizer(args, params, train_loader):

    optimizer = get_optimizer(params, optim_type=args.optim, wd=args.wd)
    # check if train_loader is tuple
    if isinstance(train_loader, tuple):
        # total_iter = len(train_loader[-1]) * args.epochs # this is for various mixing methods
        total_iter = len(train_loader[0]) * args.epochs # this is for various mixing methods
    else:
        total_iter = len(train_loader) * args.epochs
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iter, eta_min=1e-9)
    warmup_lr = 1e-5 if args.lr_backbone > 5e-5 else 1e-6
    scheduler = get_warmup_scheduler(optimizer=optimizer, scheduler=base_scheduler, warmup_iter=50, warmup_lr=warmup_lr)

    return optimizer, scheduler, total_iter


def set_params(args, model, classifier_head, logger, dataset_classifier_head=None):

    params_classifier = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]
    params_visual = [{'params': model.visual.parameters(), 'lr': args.lr_backbone}]
    # params_transformer = [{'params': model.transformer.parameters(), 'lr': args.lr_backbone}]
    if dataset_classifier_head is not None:
        params_dataset_classifier = [{'params': dataset_classifier_head.parameters(), 'lr': args.lr_classifier}]

    if args.method == "zeroshot":
        logger.info('zeroshot only.')
        for param in model.parameters():
            param.requires_grad = False
        params = params_classifier # place holder
        logit_scale = torch.tensor([4.60517]).to(device=args.device)

    elif args.method == "probing" or args.method == "REAL-Linear" or args.method == "CMLP":
        logger.info('Freezing the visual encoder.')
        for param in model.parameters():
            param.requires_grad = False
        params = params_classifier
        logit_scale = torch.tensor([4.60517]).to(device=args.device)
        # 4.60517 = np.log(100) = np.log(1 / 0.01), 0.01 is the temperature

    elif args.method == "finetune" or args.method == "finetune-multitask" or \
        args.method == "finetune-mixed" or args.method == "dataset-cls" or \
        args.method == "mixup" or args.method == "mixup-fs" or \
        args.method == "cutmix" or args.method == "cutmix-fs" or args.method == "resizemix" or \
        args.method == "saliencymix" or args.method == "attentivemix" or \
        args.method == "CMO" or args.method == "fixmatch":

        logger.info('Training the visual encoder and linear head.')

        for param in model.parameters():
            param.requires_grad = True

        lock_text_tower(model)
        params = params_classifier + params_visual
        # params = params_visual # for ablating stage 2 with frozen classifier

        if args.method == "finetune-multitask":
            params = params + params_dataset_classifier

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature)) # ln(1/0.07)=2.65926
        params.append({'params': [logit_scale], 'lr': args.lr_classifier})

    elif args.method == "FLYP":
        logger.info('Training the visual encoder and text encoder.')

        for param in model.parameters():
            param.requires_grad = True

        if args.locked_text:
            logger.info('Freezing the text encoder.')
            # from REACT's lock_text_tower()
            lock_text_tower(model)
            # params = params_visual
        """
        # check if the model.visual.proj and model.text_projection are in the params
        for name, param in model.named_parameters():
            print(name)
        """

        if args.lr_projector is None:
            args.lr_projector = args.lr_backbone

        # here we first set the projectors to be frozen to exclude them from the params list
        model.visual.proj.requires_grad = False
        model.text_projection.requires_grad = False

        total_params = list(model.parameters())
        params_list = [p for p in total_params if p.requires_grad]
        params = [{'params': params_list, 'lr': args.lr_backbone}]

        # set a different learning rate for the model.visual.proj and model.text_projection
        model.visual.proj.requires_grad = True
        model.text_projection.requires_grad = True
        params_visual_proj = {'params': [model.visual.proj], 'lr': args.lr_projector}
        params.append(params_visual_proj)
        params_text_proj = {'params': [model.text_projection], 'lr': args.lr_projector}
        params.append(params_text_proj)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature)) # ln(1/0.07)=2.65926
        params.append({'params': [logit_scale], 'lr': args.lr_classifier})

    else:
        raise NotImplementedError(f'Method {args.method} not implemented.')

    args.logit_scale = logit_scale

    return params, logit_scale



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


