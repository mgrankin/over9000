# the code mostly from https://github.com/sdoria/SimpleSelfAttention

# adapted from https://github.com/fastai/fastai/blob/master/examples/train_imagenette.py
# added self attention parameter
# changed per gpu bs for bs_rat


from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
#from fastai.vision.models.xresnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
from xresnet import *
from functools import partial
import statsmodels.stats.api as sms

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_data(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

from radam import *
from novograd import *
from ranger import *
from ralamb import *
from over9000 import *
from lookahead import *
from lamb import *
from diffgrad import DiffGrad
from adamod import AdaMod

def fit_with_annealing(learn:Learner, num_epoch:int, lr:float=defaults.lr, annealing_start:float=0.7)->None:
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr)
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(num_epoch)
    
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
        mixup: Param("Mixup", float)=0.,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50)", str)='xresnet50',
        sa: Param("Self-attention", int)=0,
        sym: Param("Symmetry for self-attention", int)=0,
        dump: Param("Print model; don't train", int)=0,
        lrfinder: Param("Run learning rate finder; don't train", int)=0,
        log: Param("Log file name", str)='log',
        sched_type: Param("LR schedule type", str)='one_cycle',
        ann_start: Param("Mixup", float)=-1.0,
        ):
    "Distributed training of Imagenette."
    
    bs_one_gpu = bs
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam' : opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd' : opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    elif opt=='ranger'  : opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb'  : opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='over9000'  : opt_func = partial(Over9000,  betas=(mom,alpha), eps=eps)
    elif opt=='lookahead'  : opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
    elif opt=='lamb'  : opt_func = partial(Lamb, betas=(mom,alpha), eps=eps)
    elif opt=='diffgrad'  : opt_func = partial(DiffGrad, version=1, betas=(mom,alpha),eps=eps)
    elif opt=='adamod'  : opt_func = partial(AdaMod, betas=(mom,alpha), eps=eps, beta3=0.999)
   
    data = get_data(size, woof, bs)
    bs_rat = bs/bs_one_gpu   #originally bs/256
    if gpu is not None: bs_rat *= max(num_distrib(), 1)
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat

    m = globals()[arch]
    
    log_cb = partial(CSVLogger,filename=log)
    
    learn = (Learner(data, m(c_out=10, sa=sa, sym=sym), wd=1e-2, opt_func=opt_func,
             metrics=[accuracy,top_k_accuracy],
             bn_wd=False, true_wd=True,
             loss_func = LabelSmoothingCrossEntropy(),
             callback_fns=[log_cb])
            )
    print(learn.path)
    
    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`
    
    if lrfinder:
        # run learning rate finder
        IN_NOTEBOOK = 1
        learn.lr_find(wd=1e-2)
        learn.recorder.plot()
    else:
        if sched_type == 'one_cycle': 
            learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
        elif sched_type == 'flat_and_anneal': 
            fit_with_annealing(learn, epochs, lr, ann_start)
    
    return learn.recorder.metrics[-1][0]

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
        mixup: Param("Mixup", float)=0.,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50)", str)='xresnet50',
        sa: Param("Self-attention", int)=0,
        sym: Param("Symmetry for self-attention", int)=0,
        dump: Param("Print model; don't train", int)=0,
        lrfinder: Param("Run learning rate finder; don't train", int)=0,
        log: Param("Log file name", str)='log',
        sched_type: Param("LR schedule type", str)='one_cycle',
        ann_start: Param("Mixup", float)=-1.0,
        ):

    acc = np.array(
        [train(gpu,woof,lr,size,alpha,mom,eps,epochs,bs,mixup,opt,arch,sa,sym,dump,lrfinder,log,sched_type,ann_start)
                for i in range(run)])
    
    print(acc)
    print(f'mean = {np.mean(acc)}, std = {np.std(acc)}, ci-95 = {sms.DescrStatsW(acc).tconfint_mean()}')

