import torch
import torchopt
from torch import optim
from torch.nn.functional import mse_loss
import pytorch_lightning as pl
from functools import partial
from functorch import vmap, make_functional, grad

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, LatentModulation

class ImplicitNeuralField(pl.LightningModule):
    """Implements a generic task implicit neural fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 module: ImplicitNeuralModule,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super().__init__()
        self.save_hyperparameters(ignore = 'module')
        self.module = module

    def initialize_modulation(self):
        m = LatentModulation(self.module.mod,
                             self.device)
        return make_functional(m)

    def initialize_inner_opt(self, mparams):
        lr = self.hparams.lr_inner
        opt = torchopt.sgd(lr=lr)
        opt_state = opt.init(mparams)
        return (opt, opt_state)

    def pred_loss(self, qs: Tensor, ys: Tensor, pred_ys: Tensor):
        return mse_loss(pred_ys, ys)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()   # average loss of all modulations
        optimizer.step()  # outer optimizer
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # each task in the batch is a group of queries and outputs
        qs, ys = batch
        # Fitting modulations for current generation
        # In parallel, trains one mod per task.
        vloss = vmap(partial(inner_modulation_loop, self))

        # fit modulations on batch - returns averaged loss
        # Compute the maml loss by summing together the returned losses.
        mod_losses = torch.mean(vloss(qs, ys))
        self.log('loss', mod_losses.item())
        return mod_losses # overriding `backward`. See above

    def fit_modulation(self, qs:Tensor, ys:Tensor):
        return fit_modulation(self, qs, ys)

    def eval_modulation(self, m, qs:Tensor):
        return eval_modulation(self, m, qs)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.module.theta.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]


def eval_modulation(exp, mod, qs : Tensor):
    (mfunc, mparams) = mod
    phi = mfunc(mparams)
    return exp.module(qs, phi)

# https://github.com/metaopt/torchopt/blob/main/examples/FuncTorch/maml_omniglot_vmap.py
# borrowed from above
def fit_modulation(exp, qs: Tensor, ys: Tensor,
                   inner_steps = None):

    # modulation in functorch form
    (mfunc, mparams) = exp.initialize_modulation()
    # init inner loop optimizer
    opt, opt_state = exp.initialize_inner_opt(mparams)

    def compute_loss(mparams):
        # using updated params
        m = (mfunc, mparams)
        pred = eval_modulation(exp, m, qs)
        pred_loss = exp.pred_loss(qs, ys, pred)
        l2_loss = torch.sum(mparams[0] ** 2)
        return pred_loss + l2_loss

    steps = exp.hparams.inner_steps if inner_steps is None else inner_steps
    new_mparams = mparams
    for _ in range(steps):
        grads = grad(compute_loss)(new_mparams)
        updates, opt_state = opt.update(grads, opt_state,
                                        inplace=False)
        new_mparams = torchopt.apply_updates(new_mparams, updates,
                                             inplace=False)

    return (mfunc, new_mparams)

def fit_and_eval(exp, qs:Tensor, ys:Tensor):
    m = fit_modulation(exp, qs, ys)
    pred = eval_modulation(exp, m, qs)
    return pred

def inner_modulation_loop(exp, qs: Tensor, ys: Tensor):
    m = fit_modulation(exp, qs, ys)
    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    pred = eval_modulation(exp, m, qs)
    pred_loss = exp.pred_loss(qs, ys, pred)
    return pred_loss
