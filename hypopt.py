import torch
from torch import nn
from collections import defaultdict, deque
import operator


class Optimizable(nn.Module):
    """
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability (see ModuleWrapper below to convert a Module to an
    Optimizable).

    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(..., optimizer=...)
        o.initialize()
        loop:
            o.retain_grad()
            o.zero_grad()
            loss = [compute loss function from parameters]
            loss.backward()
            o.adjust()

    Optimizables recursively handle updates to their optimiz*ers*.
    """

    def __init__(self, optimizer=True):
        super().__init__()
        self.optimizer = NoOpOptimizer() if optimizer else None

    def retain_grad(self):
        """Enable gradient tracking on current parameters."""
        for name, param in self.named_parameters(recurse=True):
            # print(name)
            if not param.requires_grad:
                # print(f'Weird: {name} didn\'t require grad...')
                param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...

    def zero_grad(self):
        """Set all gradients to zero."""
        # Module should be able to zero the grads for us
        super().zero_grad()
        # However annoyingly keeps grad=None on "new" parameters,
        # so we force them to materialize manually, saving us having
        # to check for None in the acctual code.
        for param in self.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param, dtype=param.dtype)

    def initialize(self, parameters=None):
        pass

    def adjust(self):
        """ Update parameters """
        pass

    def __truediv__(self, other: "Optimizable"):
        # Slightly clupsy way of making division right-associative...
        self.optimizer = self.optimizer / other
        return self


class NoOpOptimizer(Optimizable):
    """
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    """

    def __init__(self):
        super().__init__(optimizer=False)  # prevent infinite recursion

    def adjust(self, params):
        return {}

    def __truediv__(self, other: "Optimizable"):
        return other

    def __repr__(self):
        return "static"


class ModuleWrapper(Optimizable):
    """
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    """

    def update_params(module, update):
        for k, v in update.items():
            *path, kk = k.split(".")
            m = module
            for sm in path:
                m = m._modules[sm]
            m._parameters[kk] = v

    def __init__(self, module):
        super().__init__()
        self.inner = module

    def initialize(self):
        update = self.optimizer.initialize(self.inner.named_parameters(recurse=True))
        if update is not None:
            ModuleWrapper.update_params(self.inner, update)
        #self.zero_grad()
        #self.adjust()
        #ModuleWrapper.update_params(self.inner, dict(self.inner.named_parameters(recurse=True)))

    def adjust(self):
        update = self.optimizer.adjust(self.inner.named_parameters(recurse=True))
        # when using recurse=True, the parameter names will be
        # on the form module0.module1.*.param_name, where module0, module1 etc.
        # are the nested modules inside self.inner.
        # We need to recover the right module for each parameter and update its
        # parameters.
        ModuleWrapper.update_params(self.inner, update)


class OptimizerWrapper(Optimizable):
    """Unfortuantely we can't hyper-optimize a black-box optimizer, but we can
    at least use it to optimize our other Optimizables."""

    def __init__(self, optimizer_class, **kwargs):
        super().__init__()
        self.opt = None
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self.debug = kwargs.pop('debug', False)

    def initialize(self, params):
        params = list(params)

        self.opt = self.optimizer_class([v for k, v in params], **self.kwargs)

        if self.debug:
            self.zero_grad()
            for k, p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            #self.opt.step()
            #self.adjust(params)
            for k, p in params:
                print(float(p.max()))
        #return dict(params)

    def adjust(self, params):
        #if self.opt is None:
        #    self.opt = self.optimizer_class([v for k, v in params], **self.kwargs)
        self.opt.step()
        #self.double()

        if self.debug:
            #params = list(params)
            for group in self.opt.param_groups:
                for p in group['params']:
                    state = self.opt.state[p]
                    if state['step'] < 3:
                        #print(1-state['exp_avg'].max()/p.grad.max())
                        print(f'OW', state['step'], f"{p.grad.max()}, {state['exp_avg'].max()}, {state['exp_avg_sq'].max()}")
        # We make sure not to create new nodes for parameters when using OptimizerWrapper,
        # since the inner optimizer only gets the references once in the initialize method.
        return dict(params)

    def __repr__(self):
        if self.opt is None:
            return f"{self.optimizer_class}({self.kwargs})"
        return f"wrapped({self.opt})"


class SGD(Optimizable):
    """
    A hyperoptimizable SGD
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def initialize(self, params):
        self.optimizer.initialize(self._parameters.items())

    def adjust(self, params) -> dict:
        # First optimize our own parameters.
        update = self.optimizer.adjust(self._parameters.items())
        # This automatically updates self.alpha, if updated
        self._parameters.update(update)
        # Warning: Updating into __dict__ like this doesn't work, since
        # the necessary nn.Parameter magic isn't run
        # self.__dict__.update({k:nn.Parameter(v) for k,v in update.items()})

        res = {}
        # Then optimize our "parent" Optimizable
        for name, param in params:
            # From paper:
            # in order to have backpropagation deposit the gradient with respect to 𝛼𝑖 as well as 𝑤𝑖,
            # we can simply refrain from detaching 𝛼𝑖 from the graph, detaching instead its parents.
            # In order to keep the computation graph clear and the detachments explicit,
            # we will also update weights by creating fresh nodes rather than changing them in-place.
            g = param.grad.detach()

            res[name] = param.detach() - self.alpha * g
        return res

    def __repr__(self):
        return f"sgd({self.alpha:.3e}) / {self.optimizer}"


class HyperSGD(Optimizable):
    def __init__(self, alpha=0.01, kappa=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.kappa = kappa

    def adjust(self, params) -> dict:
        # If this is the first call on this optimizer, alpha.grad will be all zeros.
        # But then thos is just a no-op, so all is good.
        da = self.alpha.grad.detach()
        self.alpha = nn.Parameter(self.alpha.detach() - self.kappa * da)

        # Optimize all parent parameters
        return {k: p.detach() - self.alpha * p.grad.detach() for k, p in params}

    def __repr__(self):
        return f"hyper-sgd(a={self.alpha:.3e}, k={self.kappa:.3e})"


class SGDPerParam(Optimizable):
    """
    SGD Per Param
    """

    def __init__(self, alpha=0.01, only_named_tensors=True, debug=False):
        """If only_named_tensors is True, each named tensor will only
        have one step-size. Otherwise the step size for a tensor will
        be an equivalently sized tensor."""
        super().__init__()
        self.default_alpha = alpha
        self.alphas = EscapedParameterDict()
        self.only_named_tensors = only_named_tensors
        self.debug = debug
        self.step = 0

    def initialize(self, params):
        # We create the parameters in self.alphas before the first
        # optimizer-recursion, so we can get a gradient all the way through
        # the chain from the beginning.
        for name, param in params:
            zero = (
                torch.tensor(0.0)
                if self.only_named_tensors
                else torch.zeros_like(param)
            )
            self.alphas[name] = zero + self.default_alpha
        self.optimizer.initialize(self.alphas.items())

    def adjust(self, params) -> dict:
        # It's weird, but we have to write into alphas._parameters instead of alphas directly.
        # This is only because writing directly into alphas requires us to first wrap the
        # tensors in a new nn.Parameter, which somehow detaches our gradient.
        update = self.optimizer.adjust(self.alphas.items())
        if self.debug:
            self.step += 1
            if self.step <= 4:
                for k, v in update.items():
                    up = v - self.alphas[k]
                    print(f"PP_SGD: Updating [{k}] by {up}. Step {self.step}")
        self.alphas.update(update)

        res = {}
        for name, param in params:
            alpha = self.alphas[name]
            g = param.grad.detach() if param.grad is not None else 0

            if self.debug:
                up = param.detach() - self.alphas[name] * g
                print(f'PP_SGD: Updating [{name}] by {up}. Step {self.step}')

            res[name] = param.detach() - alpha * g
        return res

    def __repr__(self):
        return f"sgd_pp(as.mean()={[float(als.mean()) for als in self.alphas.values()]}, {self.only_named_tensors=}) / {self.optimizer}"


class EscapedParameterDict(nn.ParameterDict):
    def escape(self, k):
        return k.replace('.', ',')

    def __setitem__(self, k, v):
        #super().__setitem__(self.escape(k), nn.Parameter(v))
        self._parameters[self.escape(k)] = v

    def __getitem__(self, k):
        return super().__getitem__(self.escape(k))

    def update(self, kvs:dict):
        # Note: currently update(...) and items() use escaped names,
        # whereas setitem and getitem use unescaped names.
        self._parameters.update(kvs)


class Hyper2SGD(Optimizable):
    """Manually differentiated HyperSGD "per parameter" """

    def __init__(self, alpha=0.01, kappa=0.01, debug=False):
        super().__init__()
        self.kappa = kappa
        self.default_alpha = alpha
        self.alphas = EscapedParameterDict()
        self.old_grads = EscapedParameterDict()
        self.debug = debug
        self.step = 0

    def initialize(self, params):
        for k, p in params:
            self.old_grads[k] = torch.zeros_like(p)
            self.alphas[k] = torch.tensor(float(self.default_alpha))

    def adjust(self, params) -> dict:
        self.step += 1
        res = {}
        for k, p in params:
            g = p.grad.detach()
            og = self.old_grads[k].detach()
            alpha = self.alphas[k].detach()
            if self.debug and self.step <= 4:
                up = self.kappa * (g.reshape(-1) @ og.reshape(-1))  # Note +
                print(f"H2SGD: Updating alpha[{k}] by {up}. Step {self.step}")
            alpha += self.kappa * (g.reshape(-1) @ og.reshape(-1))  # Note +
            self.alphas[k] = alpha
            self.old_grads[k] = g

            res[k] = p.detach() - alpha * g
        return res

    def __repr__(self):
        return f"hyper2-sgd(as={[float(v) for v in self.alphas.values()]}, k={self.kappa:.3e})"


class Hyper3SGD(Optimizable):
    """Manually differentiated HyperHyperSGD "per parameter" """

    def __init__(self, alpha=0.01, kappa=0.01, gamma=0.01):
        super().__init__()
        self.gamma = gamma
        # TODO: Move to ParameterDict, if cuda is going to work
        self.kappas = defaultdict(lambda: kappa)
        self.alphas = defaultdict(lambda: alpha)
        self.old1_grads = {}
        self.old2_grads = {}

    def initialize(self, params):
        for k, p in params:
            self.old1_grads[k] = torch.zeros_like(p)
            self.old2_grads[k] = torch.zeros_like(p)

    def adjust(self, params) -> dict:
        res = {}
        for k, p in params:
            g = p.grad.detach()
            o1g = self.old1_grads[k]
            o2g = self.old2_grads[k]

            kappa = self.kappas[k] + self.gamma * (g * o1g).sum() * (o1g * o2g).sum()
            self.kappas[k] = kappa

            alpha = self.alphas[k] + kappa * (g * o1g).sum()
            self.old2_grads[k] = o1g
            self.alphas[k] = alpha

            res[k] = p.detach() - alpha * g
            self.old1_grads[k] = g
        return res

    def __repr__(self):
        return f"hyper3-sgd(as={[float(v) for v in self.alphas.values()]}, ks={[float(v) for v in self.kappas.values()]}, g={self.gamma:.3e})"


class ParameterDeque(nn.Module):
    def __init__(self) -> None:
        super(ParameterDeque, self).__init__()
        self.left = 0
        self.right = 0 # Points at the first non-existing element

    def _convert_idx(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        idx += self.left
        return str(idx)

    def __getitem__(self, idx):
        return self._parameters[self._convert_idx(idx)]

    def __setitem__(self, idx: int, param):
        self._parameters[self._convert_idx(idx)] = param

    def append(self, param):
        self._parameters[str(self.right)] = param
        self.right += 1

    def __len__(self) -> int:
        return self.right - self.left

    def __iter__(self):
        return (self._parameters[str(i)] for i in range(self.left, self.right))

    def __delitem__(self, idx):
        sidx = self._convert_idx(idx)
        if sidx == str(self.left):
            self._parameters[sidx] = None
            self.left += 1
        elif sidx == str(self.right-1):
            self.right -= 1
        else:
            raise IndexError('Can only delete from front and back.')


class InfiniteSGD(Optimizable):
    """Same as having an infinite tower of SGD's()"""

    def __init__(self, default_lr=0.01, max_levels=None, clip=None, vclip=None):
        super().__init__()
        self.default_lr = torch.tensor(default_lr)
        self.vs = nn.ModuleDict()
        self.lrs = nn.ModuleDict()
        self.old_grad = EscapedParameterDict()
        if max_levels is not None:
            assert max_levels >= 2
        self.max_levels = max_levels
        self.clip = clip # Gradient clipping
        self.vclip = vclip
        self.step  = 0

    def initialize(self, params):
        for name, param in params:
            cname = name.replace('.', ',')
            self.old_grad[cname] = torch.zeros_like(param).reshape(-1)
            self.vs[cname] = ParameterDeque()
            self.lrs[cname] = ParameterDeque()
            self.lrs[cname].append(self.default_lr)

    def adjust(self, params) -> dict:
        res = {}
        for name, param in params:
            cname = name.replace('.', ',')
            g = param.grad.detach()

            if self.clip is not None:
                gn = g.norm()
                if gn > self.clip:
                    g = (g / gn) * self.clip

            v = g.reshape(-1) @ self.old_grad[cname]

            # Gradient clipping
            if self.vclip is not None and abs(v) > self.vclip:
                v = torch.sign(v) * self.vclip

            self.old_grad[cname] = g.reshape(-1)
            vs, lrs = self.vs[cname], self.lrs[cname]

            vs.append(v)
            # vs[-2] *= vs[-1]
            # vs[-3] *= vs[-2]
            # ...
            # vs[0] *= vs[1]
            for i in range(len(vs)-1):
                vs[-i-2] = vs[-i-2] * vs[-i-1]
                # Taking the square-root here is another way to avoid
                # v-explosion. But does it make any sense?
                #if self.vclip is not None:
                    #vs[-i-2] /= vs[-i-2].abs().sqrt()

            lrs.append(self.default_lr)
            # lrs[-2] += lrs[-1]*vs[0]
            # lrs[-3] += lrs[-2]*vs[1]
            # ...
            # lrs[0] += lrs[1]*vs[-1] # lr += kappa * v
            for i in range(len(lrs)-1):
                # Negative learning rates are just ugly to me, but maybe
                # it's my predudice?
                lrs[-i-2] = max(torch.tensor(0.), lrs[-i-2] + lrs[-i-1]*vs[i])

            # Hyper3Sgd needs to save 1 v value. Hyper2Sgd needs none.
            if self.max_levels is not None and len(vs)+1 >= self.max_levels:
                del vs[0]
                del lrs[-1]

            res[name] = param.detach() - lrs[0].detach() * g
        return res

    def __repr__(self):
        lrs = {k: ', '.join(f'{lr:.3e}' for lr in list(lrs)[:5]) for k, lrs in self.lrs.items()}
        vs = {k: ', '.join(f'{v:.3e}' for v in list(vs)[-5:]) for k, vs in self.vs.items()}
        return f"inf_sgd({lrs=}, {vs=}) / {self.optimizer}"


class SGD_WithMatrix(Optimizable):
    """
    Will SGD Learn the Hessian?
    """

    def __init__(self, alpha=0.01, row_wise=True):
        super().__init__()
        self.default_alpha = alpha
        self.row_wise = row_wise
        self.alphas = EscapedParameterDict()

    def initialize(self, params):
        for name, param in params:
            size = (
                param.shape[0]
                if self.row_wise
                else torch.prod(torch.tensor(param.shape))
            )
            self.alphas[name] = (1 - 2 * torch.rand((size, size))) * self.default_alpha
        self.optimizer.initialize(self.alphas.items())

    def adjust(self, params) -> dict:
        self.alphas.update(self.optimizer.adjust(self.alphas.items()))

        res = {}
        for name, param in params:
            g = param.grad.detach()
            if self.row_wise:
                x = param.detach().reshape(param.shape[0], -1)
                g = g.reshape(param.shape[0], -1)
            else:
                x = param.detach().reshape(-1)
                g = g.reshape(-1)
            res[name] = (x - self.alphas[name] @ g).reshape(param.shape)
        return res

    def __repr__(self):
        return f"sgd_matrix(as.mean()={[float(als.mean()) for als in self.alphas.values()]}) / {self.optimizer}"


class SGD_WithRNN(Optimizable):
    """
    Will the RNN learn momentum?
    """

    def __init__(self, alpha=1, **rnn_args):
        super().__init__()
        self.alpha = alpha
        self.num_layers = rnn_args.get("num_layers", 1)
        if not "nonlinearity" in rnn_args:
            rnn_args["nonlinearity"] = "relu"
        self.rnn_args = rnn_args
        self.rnns = nn.ModuleDict()
        self.hiddens = nn.ParameterDict()

    def adjust(self) -> dict:
        update = self.optimizer.adjust(self.rnns.named_parameters())
        ModuleWrapper.update_params(self.rnns, update)

        res = {}
        for name, param in params:
            cname = name.replace(".", ",")
            if cname not in self.rnns:
                size = torch.prod(torch.tensor(param.shape))
                self.rnns[cname] = nn.RNN(size, size, **self.rnn_args)
                # It's OK to write directly into this ParameterDict (rather than using hiddens._parameters)
                # since we don't optimizer over the hidden vectors.
                self.hiddens[cname] = nn.Parameter(
                    torch.zeros(self.num_layers, 1, size)
                )

            g = param.grad.detach().reshape(1, 1, -1)
            new_g, h1 = self.rnns[cname](g, self.hiddens[cname])
            self.hiddens[cname] = nn.Parameter(h1)
            res[name] = param.detach() - self.alpha * new_g.reshape(param.shape)
        return res

    def __repr__(self):
        return f"rnn() / {self.optimizer}"
        parts = []
        for k, r in self.rnns.items():
            parts.append(f"{k}: {r.weight_ih_l.norm()=}, {r.weight_hh_l.norm()=}")
        return f"rnn({'; '.join(parts)}) / {self.optimizer}"


class Adam(Optimizable):
    """
    A fully hyperoptimizable Adam optimizer
    """

    def clamp(x):
        return (x.tanh() + 1.0) / 2.0

    def unclamp(y):
        z = y * 2.0 - 1.0
        return ((1.0 + z) / (1.0 - z)).log() / 2.0

    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        log_eps=-8.0,
        fudge=1e-14,
        optparams=None,
        debug=False
    ):
        super().__init__()
        self.num_adjustments = 0
        self.debug = debug

        self.alpha = nn.Parameter(torch.tensor(alpha))
        # The betas must be in [0, 1]. To avoid this issue in training,
        # we map them into "logits" in (-inf, inf).
        self.beta1 = nn.Parameter(Adam.unclamp(torch.tensor(beta1)))
        self.beta2 = nn.Parameter(Adam.unclamp(torch.tensor(beta2)))
        self.log_eps = nn.Parameter(torch.tensor(log_eps))

        # dicts are used to store the momentum and velocity for each parameter
        # we are training. You may ask: Why store this in a ParameterDict,
        # rather than a normal dict, since we are not going to train those parameters?
        # The answer is that using a ParamterDict allows nn.Module to find the
        # parameters when we need to .zero_grad() and .cuda() them.
        self.exp_avgs = EscapedParameterDict()
        self.exp_avg_sqs = EscapedParameterDict()

        # If optparams is None, optimize everything
        self.optparams = optparams or ("alpha", "beta1", "beta2", "log_eps")

        # We add a little 'fudge factor' to the exp_avg_sqs (+eps) because the square-root
        # is not differentiable at 0. This is not in PyTorch's implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py#L115
        # but we seem to need it when stacking Adams.
        # Set this to 0 when comparing simple outputs with pytorch's Adam.
        self.fudge = fudge

    def initialize(self, params):
        params = list(params)

        for name, param in params:
            self.exp_avgs[name] = torch.zeros_like(param)
            self.exp_avg_sqs[name] = torch.zeros_like(param) + self.fudge
        self.optimizer.initialize(((k, self._parameters[k]) for k in self.optparams))


        if self.debug:
            self.zero_grad()
            for k, p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            self.adjust(params)
            self.num_adjustments -= 1
            for k, p in params:
                print('pm', float(p.max().data))

    def adjust(self, params):
        # First adjust our own parameters
        self._parameters.update( self.optimizer.adjust(((k, self._parameters[k]) for k in self.optparams)))
        #self._parameters.update({})

        # Then do the normal Adam optimization of our parent's parameters
        self.num_adjustments += 1
        beta1 = Adam.clamp(self.beta1)
        beta2 = Adam.clamp(self.beta2)
        patch = {}
        for name, param in params:
            # The rule is: Everything but our own parameters (alpha, beta, etc.)
            # is detached.
            g = param.grad.detach()
            m = self.exp_avgs[name].detach()
            v = self.exp_avg_sqs[name].detach()

            # Update m and v with momentum.
            # m(t+1) = b m(t) + (1-b) g
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g.conj())

            if self.debug and self.num_adjustments < 3:
                #print(1-m.max()/g.max())
                print(self.num_adjustments, f'{g.max()}, {m.max()}, {v.max()}')

            # The fresh (cut off) m and v are stored back into the module
            #self.exp_avgs[name] = nn.Parameter(m)
            #self.exp_avg_sqs[name] = nn.Parameter(v)
            self.exp_avgs[name] = m
            self.exp_avg_sqs[name] = v



            # Bias correction.
            m_hat = m / (1.0 - beta1 ** self.num_adjustments)
            v_hat = v / (1.0 - beta2 ** self.num_adjustments)

            dparam = m_hat / (v_hat.sqrt() + 10.0 ** self.log_eps)
            patch[name] = param.detach() - self.alpha * dparam
        return patch

    def __repr__(self):
        return f"Adam(a={self.alpha:.3e}, betas=({Adam.clamp(self.beta1):.3f}, {Adam.clamp(self.beta2):.3f}), eps={10**self.log_eps:.3e}) / {self.optimizer}"


class FixedAdam(Optimizable):
    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        log_eps=-8.0,
        fudge=1e-14,
        weight_decay=0.0,
        amsgrad=False,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta1 = nn.Parameter(Adam.unclamp(torch.tensor(beta1)))
        self.beta2 = nn.Parameter(Adam.unclamp(torch.tensor(beta2)))
        self.beta1_prod = 1
        self.beta2_prod = 1
        self.log_eps = nn.Parameter(torch.tensor(log_eps))
        self.fudge = fudge
        self.weight_decay = nn.Parameter(torch.tensor(weight_decay))
        self.amsgrad = amsgrad
        self.exp_avgs = EscapedParameterDict()
        self.exp_avg_sqs = EscapedParameterDict()
        self.max_exp_avg_sqs = EscapedParameterDict()

    def initialize(self, params):
        for name, param in params:
            self.exp_avgs[name] = nn.Parameter(torch.zeros_like(param))
            self.exp_avg_sqs[name] = nn.Parameter(
                torch.zeros_like(param) + self.fudge
            )
            if self.amsgrad:
                self.max_exp_avg_sqs[name] = nn.Parameter(
                    torch.zeros_like(param) + self.fudge
                )
        self.optimizer.initialize(self._parameters.items())

    def adjust(self, params):
        self._parameters.update(self.optimizer.adjust(self._parameters.items()))
        lr = self.alpha
        beta1 = Adam.clamp(self.beta1)
        beta2 = Adam.clamp(self.beta2)
        eps = 10 ** self.log_eps

        res = {}
        for name, param in params:
            m = self.exp_avgs[name].detach()
            v = self.exp_avg_sqs[name].detach()
            g = param.grad.detach()
            x = param.detach()

            bias_correction1 = 1 - beta1 * self.beta1_prod
            bias_correction2 = 1 - beta2 * self.beta2_prod

            if self.weight_decay != 0:
                g += x * self.weight_decay

            # Decay the first and second moment running average coefficient
            # m.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            m = m * beta1 + g * (1 - beta1)
            v = v * beta2 + (g * g.conj()) * (1 - beta2)

            # m and v are replaced by fresh nodes
            self.exp_avgs[name] = nn.Parameter(m)
            self.exp_avg_sqs[name] = nn.Parameter(v)

            if self.amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                w = self.max_exp_avg_sqs[name].detach()
                w = torch.maximum(w, v)
                self.max_exp_avg_sqs[name] = nn.Parameter(w)
                # Use the max. for normalizing running avg. of gradient
                denom = (w / bias_correction2).sqrt() + eps
            else:
                # Pytorch uses this version, with two square root operations.
                # Does that give better precision or something? I don't think so...
                # denom = (v.sqrt() / bias_correction2.sqrt()).add_(eps)
                denom = (v / bias_correction2).sqrt() + eps

            step_size = lr / bias_correction1

            # param.addcdiv_(m, denom, value=-step_size)
            res[name] = x - step_size * (m / denom)

        self.beta1_prod *= float(beta1)
        self.beta2_prod *= float(beta2)

        return res

    def __repr__(self):
        return f"Adam2(a={self.alpha:.3e}, betas=({Adam.clamp(self.beta1):.3f}, {Adam.clamp(self.beta2):.3f}), eps={10**self.log_eps:.3e}, weight_decay={self.weight_decay:.3e}, amsgrad={self.amsgrad}) / {self.optimizer}"
