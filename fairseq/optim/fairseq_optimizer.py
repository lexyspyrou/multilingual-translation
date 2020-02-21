# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import numpy as np

class FairseqOptimizer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lan_probs = None

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    def init_lan_sim(self, lan_probs):
        self.lan_probs = lan_probs
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                #if p.grad is None: continue
                if p.numel() < 10: continue
                state = self.optimizer.state[p]
                state['lan_probs'] = np.array([1./prob for prob in lan_probs])
                state['lan_probs'] = state['lan_probs'] / np.sum(state['lan_probs'])
                #state['lan_probs'] = np.array([1./len(lan_probs) for i in lan_probs])
                #state['lan_probs'] = np.array([1.0 for i in range(len(lan_probs))])
                state['lan_sim'] = np.array([1./len(lan_probs) for i in lan_probs])
                #state['normed_lan_probs'] = [1. for _ in range(len(lan_probs))]
                state['normed_lan_probs'] = state['lan_probs'] * len(state['lan_probs'])

    def save_grad_sim_for_id(self, i):
        print("save grad_sim")
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                #if p.grad is None: continue
                state = self.optimizer.state[p]
                if "lan_sim" not in state: continue 

                cosine_prod = (state['dev_grad'] * p.grad.data).sum().item()
                cosine_norm = p.grad.data.norm(2) ** 2
                dev_cosine_norm = state['dev_grad'].norm(2) ** 2
                cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
                #if grad_sim == "cosine":
                #    cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
                #elif grad_sim == "dot_prod":
                #    cosine_sim = cosine_prod
                state['lan_sim'][i] = cosine_sim.item()
                print(state['lan_sim'][i])

    def update_lan_probs(self):
        num_param = 0
        acc_probs = []
        print("update lan probs")
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                #if p.grad is None: continue
                state = self.optimizer.state[p]
                if "lan_probs" not in state: continue 
                num_param += 1
                s = 0
                if len(acc_probs) == 0:
                    acc_probs = [0. for _ in range(len(state['lan_probs']))]
                for _ in range(self.args.data_actor_optim_step):
                    log_lan_probs = np.log(state['lan_probs'] + 1e-10)

                    weighted_r = state['lan_sim'] * state['lan_probs']
                    grad = state['lan_sim'] - np.sum(weighted_r)

                    log_lan_probs += grad * self.args.data_actor_lr[0]
                    #state['lan_probs'] = np.exp(log_lan_probs)
                    ex = np.exp(log_lan_probs - np.max(log_lan_probs))
                    state['lan_probs'] = ex / ex.sum()

                for i in range(len(state['lan_sim'])):
                    acc_probs[i] += state['lan_probs'][i]

                #for i in range(len(state['lan_sim'])):
                #    if len(acc_probs) <= i: acc_probs.append(0.)
                #    state['lan_probs'][i] += state['lan_sim'][i]*self.args.data_actor_lr[0]
                #    state['lan_probs'][i] = max(state['lan_probs'][i], 0.5)
                #    state['lan_probs'][i] = min(state['lan_probs'][i], 1.5)
                #    s += state['lan_probs'][i]
                #for i in range(len(state['lan_sim'])):
                #    state['lan_probs'][i] /= s
                #    acc_probs[i] += state['lan_probs'][i]
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                #if p.grad is None: continue
                state = self.optimizer.state[p]
                if "lan_probs" not in state: continue 
                state['normed_lan_probs'] = state["lan_probs"] * len(state['lan_sim'])
                mask = state['normed_lan_probs'] < 1.
                state['normed_lan_probs'][mask] = 1.
                #for i in range(len(state['lan_sim'])):
                #   state['normed_lan_probs'][i] = state["lan_probs"][i] * num_param /acc_probs[i]
                print(state['normed_lan_probs'])

    def multiply_grad(self, i):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if 'normed_lan_probs' not in state: return
                p.grad *= state['normed_lan_probs'][i]
                #if 'lan_probs' not in state: return
                #p.grad *= state['lan_probs'][i]

    def aggregate_lan_probs(self):
        probs = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                #if p.grad is None: continue
                state = self.optimizer.state[p]
                if not 'lan_probs' in state: continue
                probs.append(state['lan_probs'])
        probs = np.sum(np.array(probs), axis=0)
        probs = probs / np.sum(probs)
        return probs


    def save_dev_grad_multi(self, utility='ave', extras=None):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    if extras == True:
                        state['dev_grad'] = p.grad.data.clone()
                    else:
                        state['dev_grad'] += p.grad.data.clone()

    def multi_dev_grad_finalize(self, utility='ave', extras=None):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    state['dev_grad'].div_(extras)
    
    def save_train_grad_id(self, i):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if 'train_grad' not in state:
                    state['train_grad'] = [None for _ in range(len(self.args.lang_pairs))]
                if 'last_grad' not in state: state['last_grad'] = 0.
                if state['train_grad'][i] is None:
                    #state['train_grad'][i] = p.grad.data.clone() - state['last_grad']
                    state['train_grad'][i] = p.grad.data.clone()
                else:
                    #state['train_grad'][i] = p.grad.data.clone()
                    #state['train_grad'][i] = self.args.a1*p.grad.data + self.args.a0*state['train_grad'][i]
                    state['train_grad'][i] = p.grad.data.clone() + state['train_grad'][i]
                #state['last_grad'] = p.grad.data.clone()
                p.grad = None

    def get_grad_sim_id(self, i, grad_sim='cosine', src_idx=None):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if src_idx is not None:
                    grad = state['train_grad'][src_idx]
                else:
                    grad = p.grad
                cosine_prod += (state['train_grad'][i] * grad.data).sum().item()
                cosine_norm += grad.data.norm(2) ** 2
                dev_cosine_norm += state['train_grad'][i].norm(2) ** 2
        if grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
            return cosine_sim.item(), cosine_norm, dev_cosine_norm
        elif grad_sim == "dot_prod":
            cosine_sim = cosine_prod 
            return cosine_sim, cosine_norm, dev_cosine_norm

    def save_train_grad(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone() - state['dev_grad']
 
    def save_proj_grad_id(self, i):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if 'train_grad' not in state:
                    state['train_grad'] = [None for _ in range(len(self.args.lang_pairs))]
                state['train_grad'][i] = p.grad.data.clone()

    def reset_train_grad_id(self, i):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['train_grad'] = [None for _ in range(len(self.args.lang_pairs))]
                state['last_grad'] = 0.


    def proj_grad_id(self, i, src_idx=None):
        for group in self.optimizer.param_groups:
           for p in group["params"]:
               if p.grad is None: continue
               state = self.optimizer.state[p]
               if state['train_grad'][i] is None: 
                   return
               elif src_idx is not None and state['train_grad'][src_idx] is None:
                   return
               else:
                   break
        if self.args.paramwise_proj_grad:
            for group in self.optimizer.param_groups:
               for p in group["params"]:
                   if p.grad is None: continue
                   state = self.optimizer.state[p]
                   if src_idx is not None:
                       grad = state['train_grad'][src_idx]
                   else:
                       grad = p.grad
                   cosine_prod = (state['train_grad'][i] * grad.data).sum().item()
                   if cosine_prod > 0: continue
                   if self.args.grad_sim == "cosine":
                       cosine_norm = grad.data.norm(2) * state['train_grad'][i].norm(2)
                       cosine_sim = cosine_prod / (cosine_norm**0.5+1e-10)
                   else:
                       cosine_sim = cosine_prod / (state['train_grad'][i].norm(2)+1e-10)
                   grad = grad - cosine_sim * state['train_grad'][i] 
        else:
            if self.args.proj_grad_sim == "cosine":
                cosine_sim, consine_norm, dev_cosine_norm = self.get_grad_sim_id(i, self.args.proj_grad_sim, src_idx)
                if cosine_sim > 0: return
            elif self.args.proj_grad_sim == "dot_prod":
                dot_prod, consine_norm, dev_cosine_norm = self.get_grad_sim_id(i, self.args.proj_grad_sim, src_idx)
                if dot_prod > 0: return
                cosine_sim = dot_prod / (dev_cosine_norm+1e-10)
            for group in self.optimizer.param_groups:
               for p in group["params"]:
                   if p.grad is None: continue
                   state = self.optimizer.state[p]
                   if src_idx is not None:
                       grad = state['train_grad'][src_idx]
                   else:
                       grad = p.grad
                   grad = grad - cosine_sim * state['train_grad'][i] 
    
    def combine_proj_grad(self):
        for group in self.optimizer.param_groups:
           for p in group["params"]:
               if p.grad is None: continue
               state = self.optimizer.state[p]
               p.grad = None
               for g in state['train_grad']:
                   if g is not None:
                       if p.grad is None:
                           p.grad = g
                       else:
                           p.grad = p.grad + g

    def save_train_grad_t0(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

    def save_train_grad(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['grad'] = p.grad.clone()

    def set_train_grad(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                if 'grad' not in state: continue
                p.grad = state['grad'].clone()

    def proj_grad(self):
        if self.args.paramwise_proj_grad:
            for group in self.optimizer.param_groups:
               for p in group["params"]:
                   if p.grad is None: continue
                   state = self.optimizer.state[p]
                   cosine_prod = (state['dev_grad'] * p.grad.data).sum().item()
                   if cosine_prod > 0: continue
                   if self.args.grad_sim == "cosine":
                       cosine_norm = p.grad.data.norm(2) * state['dev_grad'].norm(2)
                       cosine_sim = cosine_prod / (cosine_norm**0.5+1e-10)
                   else:
                       cosine_sim = cosine_prod / (state['dev_grad'].norm(2)+1e-10)
                   p.grad = p.grad - cosine_sim * state['dev_grad'] 
        else:
            if self.args.proj_grad_sim == "cosine":
                cosine_sim, consine_norm, dev_cosine_norm = self.get_grad_sim(self.args.proj_grad_sim)
                if cosine_sim > 0: return
            elif self.args.proj_grad_sim == "dot_prod":
                dot_prod, consine_norm, dev_cosine_norm = self.get_grad_sim(self.args.proj_grad_sim)
                if dot_prod > 0: return
                cosine_sim = dot_prod / (dev_cosine_norm+1e-10)
            for group in self.optimizer.param_groups:
               for p in group["params"]:
                   if p.grad is None: continue
                   state = self.optimizer.state[p]
                   p.grad = p.grad - cosine_sim * state['dev_grad'] 

    def save_dev_grad(self):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

    def clone_param(self):
        """Save a copy of the params"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                state['param_copy'] = p.clone()

    def add_grad(self, eta):
        """add grad to current param"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                p.data += state['dev_grad']*eta

    def switch_param(self, clear_cache=False):
        """Swap copy and the param values"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                cur_p = p.data
                p.data = state['param_copy']
                if clear_cache:
                    state['param_copy'] = None 
                else:
                    state['param_copy'] = cur_p

    def get_grad_sim(self, grad_sim='cosine'):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                cosine_prod += (state['dev_grad'] * p.grad.data).sum().item()
                cosine_norm += p.grad.data.norm(2) ** 2
                dev_cosine_norm += state['dev_grad'].norm(2) ** 2
        if grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
            return cosine_sim, cosine_norm, dev_cosine_norm
        elif grad_sim == "dot_prod":
            cosine_sim = cosine_prod 
            return cosine_sim, cosine_norm, dev_cosine_norm

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss, retain_graph=False):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward(retain_graph=retain_graph)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False
