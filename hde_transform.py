import torch
import torch.nn as nn
from torch.nn.functional import softplus
from pyro.distributions.torch_transform import TransformModule


class RowTransformModule(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dims,
            context_dim = 0,
            l_init = 0.0,
            param_dims=[1, 1],
            nonlinearity=nn.ReLU()):
        super(RowTransformModule, self).__init__()

        self.l_init = l_init
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.all_ones = (torch.tensor(param_dims) == 1).all().item()

        # Create masked layers
        if len(hidden_dims)>0:
            layers = [nn.Linear(input_dim + self.context_dim, hidden_dims[0])]
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.Linear(hidden_dims[-1], input_dim * self.output_multiplier))
        else:
            layers = [nn.Linear(input_dim + self.context_dim, input_dim * self.output_multiplier)]
        self.layers = nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x):
        y = torch.zeros_like(x)
        y[...,1:,:] = x[...,:-1,:]
        y[...,0,:] = self.l_init
        self.context = self.context.expand(x.size(0), -1, x.size(-2), -1)
        h = torch.cat((self.context, y), -1)
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(h.size()[:-1]) + [self.output_multiplier, self.input_dim])

            # Squeeze dimension if all parameters are one dimensional
            if self.count_params == 1:
                return h

            elif self.all_ones:
                return torch.unbind(h, dim=-2)


    def set_context(self, context):
        # assert context.size(-1) == self.context_dim, 'Context size mismatch'
        self.context = context


class OrderTransform(TransformModule):

    def __init__(self, num, dev, dim=0):
        super(OrderTransform, self).__init__(cache_size=1)
        self._cached_log_scale = None
        self.dim = dim
        self.upo = torch.triu(torch.ones(num-1, num-1)).to(dev)

    def _call(self, x):
        self._cached_log_scale = torch.zeros_like(x).cuda()
        log_scale = x[..., 1:, self.dim].clone()
        scale = softplus(log_scale)
        self._cached_log_scale[..., 1:, self.dim] = torch.log(scale)
        dif = (scale.unsqueeze(-1)*self.upo).sum(-2)
        x[..., 1:, self.dim] = x[..., 0, self.dim].unsqueeze(-1) + dif

        return x

    def log_abs_det_jacobian(self, x, y):
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            print('Wrong!')
        return log_scale.sum(-1)


class PosTransform(TransformModule):
    def __init__(self, dev, dim=0):
        super(PosTransform, self).__init__(cache_size=1)
        self._cached_log_scale = None
        self.dim = dim
        self.dev = dev

    def _call(self, x):
        scale = 0.5*torch.sigmoid(x[..., self.dim])
        log_scale = torch.log(scale*(1.0-2.0*scale))
        self._cached_log_scale = torch.zeros_like(x).to(self.dev)
        self._cached_log_scale[..., self.dim] = log_scale
        x[..., self.dim] = scale

        return x

    def log_abs_det_jacobian(self, x, y):
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            print('Wrong!')
        return log_scale.sum(-1)