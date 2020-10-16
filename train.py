import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
from functools import partial
import statsmodels.stats.api as sms
from fastai.vision.all import *
from pytorch_lightning.callbacks import LearningRateMonitor

torch.backends.cudnn.benchmark = True

def get_data(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)
    return (ImageDataLoaders.from_folder(path, valid='val', 
                item_tfms=RandomResizedCrop(size, min_scale=0.35), batch_tfms=Normalize.from_stats(*imagenet_stats)))

from radam import *
from novograd import *
from ranger import *
from ralamb import *
from rangerlars import *
from lookahead import *
from lamb import *
from diffgrad import DiffGrad
from adamod import AdaMod
from madam import Madam
from apollo import Apollo
from adabelief import AdaBelief

def d(x): 
    return 1

class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps, pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        super(ConcatLR, self).__init__(optimizer, last_epoch)
    
    def step(self):
        if self.last_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()
        
    def get_lr(self):
        if self.last_epoch <= self.step_start:
            return self.scheduler1.get_lr()
        else:
            return self.scheduler2.get_lr()

class LModel(pl.LightningModule):
    def __init__(self, model, sched_type, total_steps, ann_start):
        super(LModel, self).__init__()
        self.model = model
        self.sched_type = sched_type
        self.total_steps = total_steps
        self.ann_start = ann_start
        self.loss_func = LabelSmoothingCrossEntropy()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)
        loss = self.loss_func(pred, y)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)
        loss = self.loss_func(pred, y)
        acc = accuracy(pred, y)
        return {'val_loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean().item()
        self.result = {'avg_val_loss': avg_loss, 'avg_acc': avg_acc}
        print(self.result)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9,0.99), eps=1e-06)
        if self.sched_type == 'flat_and_anneal':
            dummy = LambdaLR(optimizer, d)
            cosine = CosineAnnealingLR(optimizer, self.total_steps*(1-self.ann_start))
            scheduler = ConcatLR(optimizer, dummy, cosine, self.total_steps, self.ann_start)
        else:
            scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=self.total_steps, pct_start=0.3,
                                                            div_factor=10, cycle_momentum=True)        
        meta_sched = {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }  
        return [optimizer], [meta_sched]

def train(
        gpu:Param("GPU to run on", str)=None,
        woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-6,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=256,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50)", str)='xresnet50',
        sched_type: Param("LR schedule type", str)='one_cycle',
        ann_start: Param("Mixup", float)=-1.0,
        ):
    "Distributed training of Imagenette."
    
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam' : opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd' : opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='ranger'  : opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb'  : opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='rangerlars'  : opt_func = partial(RangerLars,  betas=(mom,alpha), eps=eps)
    elif opt=='lookahead'  : opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
    elif opt=='lamb'  : opt_func = partial(Lamb, betas=(mom,alpha), eps=eps)
    elif opt=='diffgrad'  : opt_func = partial(DiffGrad, version=1, betas=(mom,alpha),eps=eps)
    elif opt=='adamod'  : opt_func = partial(AdaMod, betas=(mom,alpha), eps=eps, beta3=0.999)
    elif opt=='madam'  : opt_func = partial(Madam, p_scale=3.0, g_bound=10.0)
    elif opt=='apollo' : opt_func = partial(Apollo, beta=mom, eps=eps, warmup=0)
    elif opt=='adabelief' : opt_func = partial(AdaBelief, betas=(mom,alpha), eps=eps)

    dls = get_data(size, woof, bs)
    
    m = globals()[arch]()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    total_steps=len(dls[0])*epochs
    lmodel = LModel(m, sched_type, total_steps, ann_start)
    trainer = pl.Trainer(gpus=gpu, max_epochs=epochs, callbacks=[lr_monitor], precision=16)

    trainer.fit(lmodel, dls[0], dls[1])
    return lmodel.result['avg_acc']

@call_parse
def main(
        run: Param("Number of run", int)=20,
        gpu:Param("GPU to run on", str)=None,
        woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-6,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=256,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50)", str)='xresnet50',
        sched_type: Param("LR schedule type", str)='one_cycle',
        ann_start: Param("Mixup", float)=-1.0,
        ):

    acc = np.array(
        [train(gpu,woof,lr,size,alpha,mom,eps,epochs,bs,opt,arch,sched_type,ann_start)
                for i in range(run)])
    
    print(acc)
    print(f'mean = {np.mean(acc)}, std = {np.std(acc)}, ci-95 = {sms.DescrStatsW(acc).tconfint_mean()}')

