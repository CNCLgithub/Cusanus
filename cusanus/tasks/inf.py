import os
import torch
import torchopt
import functools
from torch import optim
from torch.nn.functional import mse_loss, l1_loss
import pytorch_lightning as pl
# import torchvision.utils as vutils
from functorch import vmap, make_functional_with_buffers, grad

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, LatentModulation
from cusanus.utils import RenderGeometry

from GPUtil import showUtilization as gpu_usage

class ImplicitNeuralField(pl.LightningModule):
    """Implements a generic task implicit neural fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 inr: ImplicitNeuralModule,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super().__init__()
        self.save_hyperparameters(ignore = 'inr')
        self.inr = inr

    def initialize_modulation(self):
        m = LatentModulation(self.inr.mod)
        m.to(self.device)
        m.train()
        return make_functional_with_buffers(m)

    def initialize_inner_opt(self, mparams):
        lr = self.hparams.lr_inner
        opt = torchopt.sgd(lr=lr)
        opt_state = opt.init(mparams)
        return (opt, opt_state)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()   # average loss of all modulations
        optimizer.step()  # outer optimizer
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # each task in the batch is a group of queries and outputs
        qs, ys = batch
        # Fitting modulations for current generation
        # In parallel, trains one mod per task.
        vloss = functools.partial(inner_modulation_loop,
                                  self)
        # fit modulations on batch - returns averaged loss
        # Compute the maml loss by summing together the returned losses.
        mod_losses = torch.mean(vmap(vloss)(qs, ys))
        self.log('loss', mod_losses.item())
        self.log('avg_gt_ys', torch.mean(ys.detach()).item(),
                 prog_bar=True)
        return mod_losses # overriding `backward`. See above

    def fit_modulation(self, qs:Tensor, ys:Tensor):
        return fit_modulation(self, qs, ys)

    def eval_modulation(self, m, qs:Tensor):
        return eval_modulation(self, m, qs)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.inr.theta.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]


def eval_modulation(exp, mod, qs : Tensor):
    (mfunc, mparams, mbuffers) = mod
    phi = mfunc(mparams, mbuffers)
    return exp.inr(qs, phi)

# https://github.com/metaopt/torchopt/blob/main/examples/FuncTorch/maml_omniglot_vmap.py
# borrowed from above
def fit_modulation(exp, qs: Tensor, ys: Tensor):

    # modulation in functorch form
    (mfunc, mparams, mbuffers) = exp.initialize_modulation()
    # init inner loop optimizer
    opt, opt_state = exp.initialize_inner_opt(mparams)

    ys_std = torch.std(ys)

    def compute_loss(mparams):
        # using updated params
        m = (mfunc, mparams, mbuffers)
        pred_ys = eval_modulation(exp, m, qs)
        rec_loss = torch.mean(mse_loss(pred_ys, ys))
        l2_loss = torch.sum(mparams[0] ** 2)
        return rec_loss + l2_loss

    new_mparams = mparams
    for _ in range(exp.hparams.inner_steps):
        # gpu_usage()
        # print(torch.cuda.memory_stats())
        grads = grad(compute_loss)(new_mparams)
        updates, opt_state = opt.update(grads, opt_state,
                                        inplace=False)
        new_mparams = torchopt.apply_updates(new_mparams, updates,
                                             inplace=False)

    return (mfunc, new_mparams, mbuffers)

def inner_modulation_loop(exp, qs: Tensor, ys: Tensor):
    m = fit_modulation(exp, qs, ys)
    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    pred_ys = eval_modulation(exp, m, qs)
    rec_loss = torch.mean(mse_loss(pred_ys, ys))
    return rec_loss
