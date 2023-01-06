import os
import torch
import torchopt
import functools
from torch import optim
from torch.nn.functional import mse_loss, l1_loss
import pytorch_lightning as pl
from functorch import vmap, make_functional_with_buffers, grad

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, LatentModulation

class EventConcepts(pl.LightningModule):
    """Training schema for event concepts

    TODO
    """

    def __init__(self,
                 inr: ImplicitNeuralModule,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super().__init__()
        self.save_hyperparameters(ignore = ['C', 'F', 'U', 'K'])
        self.C = C
        self.F = F
        self.U = U

    def initialize_modulation(self, d:int):
        m = LatentModulation(d)
        m.to(self.device)
        m.train() # REVIEW: is this neccessary?
        return make_functional(m)

    def initialize_emod(self):
        self.initialize_modulation(self.e_dim)

    def initialize_dmod(self):
        self.initialize_modulation(self.d_dim)

    def init_emods(self, n:int):
        f, _ = self.initialize_emod()
        params = [self.initialize_emod[1] for _ in range(n)]
        return f, params

    def init_dmods(self, n:int):
        f, _ = self.initialize_dmod()
        params = [self.initialize_dmod()[1] for _ in range(n)]
        return f, params

    def init_mod_optim(self, mparams):
        lr = self.hparams.lr_inner
        opt = torchopt.sgd(lr=lr)
        opt_state = opt.init(mparams)
        return (opt, opt_state)

    def pred_loss_inner(self, qs:Tensor, pred_ys:Tensor,
                        pvals:Tensor, kmod: Tensor):
        """Prediction loss per object"""
        alt_ys = self.K(qs, kmod)
        pred_loss = torch.sum(-torch.log((alt_ys * pvals) + \
                                         (pred_ys * (1.0 - pvals))))
        return pred_loss

    def pred_loss(self, qs, pred_ys, pvals, kmods):
        """Average loss across objects"""
        vl = vmap(self.pred_loss_inner)(qs, pred_ys,
                                        pvals, kmods)
        return torch.mean(vl)

    def extract_emods(self, params):
        # assumes batched eparams (events x edim)
        f, _ = self.initialize_emod()
        return vmap(f)(params)

    def extract_dmods(self, params):
        # assumes batched params (objects x ddim)
        f, _ = self.initialize_dmod()
        return vmap(f)(params)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()   # average loss of all modulations
        optimizer.step()  # outer optimizer
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # each task in the batch a simulation scene
        # Fitting modulations for current generation in parallel
        vloss = vmap(functools.partial(event_loop,
                                       self))
        # fit modulations on batch - returns averaged loss
        # Compute the maml loss by aggregate loss.
        loss = torch.mean(vloss(batch))
        self.log('loss', mod_losses.item())
        return loss # overriding `backward`. See above

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

def eval_event_modulation(exp, seq, dmods, pair, emod):
    (a_key, b_key) = pair
    a = seq['objects'][a_key]
    b = seq['objects'][b_key]
    a_dmod = dmods[a_key]
    b_dmod = dmods[b_key]
    return exp.event_module(a, b, emod, a_dmod, b_dmod)

def update_state(exp, seq, pvals, fmods, dmods, obj_key):
    obj = seq['objects'][obj_key]
    pairs = seq['pairs']
    dmod = dmods[obj_key]
    fmod = torch.zeros_like(fmods[0])
    pval = torch.ones(1).to(exp.device)
    for (a,b),f,p in zip(pairs, fmods, pvals):
        if a == obj_key:
            fmod += f
            pval *= p
        elif b == obj_key:
            fmod -= f
            pval *= p

    return pval, exp.update_module(obj, fmod, dmod)

def eval_event_modulations(exp, seq, dmods, emods):
    """Applieds m events to n objects"""
    eval_emod = functools.partial(eval_event_modulation,
                                  exp, seq, dmods)
    pairs = seq['pairs']
    # forces for each event
    pvals, fmods = vmap(efunc)(pairs, emods)

    update_func = functools.partial(update_state,
                                    seq, fmods,
                                    dmods)
    # updated states for each object
    # with pvals for each update
    pvals, new_ks = vmap(update_func)(obj_keys)
    return (pvals, new_ks)



def fit_event_modulations(exp, dmods, seq:dict):
    """Fits event codes for each pair of objects in a sequence"""
    efunc, eparams = exp.init_emods(len(seq['pairs']))
    eopt, eopt_state = exp.init_mod_optim(eparams)
    eval_f = functools.partial(eval_event_modulations,
                               exp, seq, dmods)
    def compute_loss(eparams):
        emods = vmap(efunc)(eparams)
        # new pvals and states
        pvals, kmods = eval_f(emods)
        qs = seq['qs']
        ys = seq['ys']
        # average loss across objects
        pred_loss = exp.pred_loss(qs, pvals, ys, kmods)
        # keep components near 0
        l2_emod = torch.sum(emods ** 2)
        # NOTE: Possible to use NFs here
        l2_kmod = torch.sum(kmods ** 2)
        return pred_loss + l2_emod + l2_kmod

    new_eparams = eparams
    for _ in range(exp.hparams.emod_steps):
        grads = grad(compute_loss)(new_eparams)
        updates, eopt_state = eopt.update(grads, eopt_state,
                                          inplace=False)
        new_eparams = torchopt.apply_updates(new_eparams, updates,
                                             inplace=False)

    return new_eparams

def eval_dynamics_modulations(exp, dmods, emods, seq):
    return eval_event_modulations(exp, seq, dmods, emods)

def loss_dynamics_modulations(exp, dmods, emods, seq):
    pvals, kmods = eval_event_modulations(exp, seq, dmods, emods)
    ys = seq['ys']
    # average loss across objects
    pred_loss = exp.pred_loss(qs, pvals, ys, kmods)
    # keep components near 0
    # NOTE: Possible to use NFs here
    l2_kmod = torch.sum(kmods ** 2)
    return pred_loss + l2_kmod

def fit_dynamics_modulations(exp, sim, eparams):
    """Fits dynamics codes to objects across a sim"""

    dfunc, dparams = exp.init_dmods(sim['nobjects'])
    dopt, dopt_state = exp.init_mod_optim(dparams)
    # only need efunc
    efunc, _ = exp.init_emods(seq)
    seqs = sim['sequences']


    def compute_loss(dparams):
        dmods = vmap(dfunc)(dparams)
        eval_f = vmap(functools.partial(loss_dynamics_modulations,
                                        exp,
                                        dmods))
        # dloss: seq x object
        dloss = eval_f(eparams, seqs)
        dloss = torch.mean(dloss)
        l2_loss = torch.sum(dmods ** 2)
        return dloss + l2_loss

    new_dparams = dparams
    for _ in range(exp.hparams.dmod_steps):
        grads = grad(compute_loss)(new_dparams)
        updates, dopt_state = eopt.update(grads, dopt_state,
                                          inplace=False)
        new_dparams = torchopt.apply_updates(new_dparams, updates,
                                             inplace=False)

    return new_dparams


def fit_sim(exp, sim:dict):
    # init dmods
    dfunc, dparams = exp.init_dmods(sim)
    dmods = vmap(dfunc, dparams)

    seqs = sim['sequences']

    for _ in range(exp.hparams.block_steps):
        # BLOCK 1: emods
        emod_f = vmap(functools.partial(
            fit_event_modulations,
            exp,
            dmods))
        eparams = emod_f(seqs)
        # seqs x nevents x edim
        emods = vmap(exp.extract_emods)(eparams)

        # BLOCK 2: dmods
        dparams = fit_dynamics_modulations(exp, sim, emods)
        # objects x ddim
        dmods = exp.extract_dmods(dparams)

    return dmods, emods


def event_loop(exp, sim:dict):
    dmods, emods = fit_sim(exp, sim)
    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    seqs = sim['sequences']
    eval_f = vmap(functools.partial(eval_dynamics_modulations,
                                    dmods, emods))
    _,  vloss = eval_f(seqs)
    pred_loss = torch.mean(vloss)
    return pred_loss
