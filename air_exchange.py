import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from pyro.nn.condi_made import ConditionalMade
from pyro.distributions import InverseAutoregressiveFlowStable
from hde_transform import OrderTransform, PosTransform
from hde_transform import RowTransformModule
from fiat import FIAT

from modules import MLP, Decoder, ConvNets, Blob2Point, GaussianSmoothing

from utils import clamp_preserve_gradients


class EAIR(nn.Module):
    def __init__(self,
                 max_obj_num,
                 x_size,
                 window_size,
                 z_what_size,
                 z_num_func,
                 decoder_net = [],
                 foc_net = [],
                 att_net = [],
                 num_lat_net = ([],[]),
                 num_obj_net = [],
                 cmade = [],
                 fmade = [],
                 decoder_output_bias=None,
                 decoder_output_use_sigmoid=False,
                 z_pres_prior = 0.01,
                 z_where_prior_loc = torch.tensor([0.3, 0.0, 0.0]),
                 z_where_prior_scale = torch.tensor([0.2, 1.0, 1.0]),
                 z_what_prior_loc = 0.0,
                 z_what_prior_scale= 1.0,
                 likelihood_sd=0.3,
                 use_cuda=False):

        super(EAIR, self).__init__()

        prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)
        self.options = dict(dtype=prototype.dtype, device=prototype.device)

        self.z_where_size = 3
        self.z_what_size = z_what_size
        self.z_prop_size = self.z_where_size + self.z_what_size
        self.max_obj_num = max_obj_num

        self.z_num_prior = z_num_func(z_pres_prior, self.max_obj_num).to(self.options['device'])
        self.z_where_prior_loc = z_where_prior_loc.to(self.options['device'])
        self.z_where_prior_scale = z_where_prior_scale.to(self.options['device'])
        self.z_what_prior_loc = z_what_prior_loc * torch.ones(z_what_size, **self.options)
        self.z_what_prior_scale = z_what_prior_scale * torch.ones(z_what_size, **self.options)

        nl1 = nn.ReLU
        nl2 = nn.Tanh
        nl3 = nn.ELU

        self.window_size = window_size
        self.decode = Decoder(window_size ** 2, decoder_net, z_what_size, 
                              decoder_output_bias, decoder_output_use_sigmoid, nl1)
        self.likelihood_sd = likelihood_sd
        self.like_pool = (1.0/(20*20))*torch.ones((1,1,10,10), **self.options)
        self.bp = Blob2Point()
        
        self.x_size = x_size
        ix, iy = torch.meshgrid([torch.linspace(-1, 1, self.x_size, **self.options),
                                           torch.linspace(-1, 1, self.x_size, **self.options)])
        self.grid = torch.stack((iy, ix), 0)
        ix = ix[[9,19,29,39]][:,[9,19,29,39]]
        iy = iy[[9,19,29,39]][:,[9,19,29,39]]
        self.cell = torch.stack((iy, ix), 0)
        self.foc_enc = MLP(window_size **2, foc_net, nl1, False)
        # self.num_lat_enc = MLP(self.z_prop_size * self.max_obj_num, num_lat_net[0], nl1, True)
        self.num_obj_enc = MLP(self.z_prop_size + att_net[0], num_obj_net + [1], nl2, False)

        self.cmade = ConditionalMade(self.z_prop_size, cmade, att_net[0])
        # self.vmade = ConditionalMade(self.max_obj_num, [64], 256, -2)
        self.l_init = nn.Parameter(torch.zeros(self.z_prop_size).to(self.options['device']))
        self.rmade = RowTransformModule(self.z_prop_size, [128], 256, self.l_init)
        self.fmade = ConditionalMade(self.z_what_size, fmade, self.foc_enc.output_size)
        self.fiat = FIAT(self.fmade, self.foc_enc, window_size)

        haf = InverseAutoregressiveFlowStable(self.cmade)
        # vaf = InverseAutoregressiveFlowStable(self.vmade)
        raf = InverseAutoregressiveFlowStable(self.rmade)
        oaf = OrderTransform(self.max_obj_num, self.options['device'], 1)
        # paf = PosTransform(self.options['device'], 0)
        faf = InverseAutoregressiveFlowStable(self.fiat)

        self.transforms = [haf, oaf, faf]

        self.frame = torch.triu(torch.ones(max_obj_num, max_obj_num, **self.options)).expand(1, 1, -1, -1)
        self.upo = (torch.ones(max_obj_num+1, max_obj_num+1, **self.options)
                    - torch.tril(torch.ones(max_obj_num+1, max_obj_num+1, **self.options)))[:max_obj_num, :]

        self.base_dist = dist.Normal
        self.facorrect = ((torch.arange(self.max_obj_num+1) + 1).to(torch.float).lgamma()).to(self.options['device'])

        self.att_enc = ConvNets(3, 64, nl3, att_net[0])
        self.num_enc = ConvNets(3, 64, nl3, att_net[0])

        self.kl_weight = 1.0
        self.gaussian_smoother = GaussianSmoothing(1, 20, 3, 10)

        base_dist = self.base_dist(
            torch.zeros(1, 1, self.max_obj_num, self.z_prop_size, **self.options),
            torch.ones(1, 1, self.max_obj_num, self.z_prop_size, **self.options))
        self.epsilon = base_dist.sample()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

        self.ims = []
        self.count = 0
        self.flag = 0

    def att_coding(self, data):
        dun_squeeze = data.unsqueeze(1)
        # xv, yv = torch.meshgrid([torch.linspace(-1, 1, self.x_size).cuda(), torch.linspace(-1, 1, self.x_size).cuda()])
        inp = torch.cat((dun_squeeze, self.grid[None,...]), 1)
        att_code = self.att_enc(inp)
        return inp, att_code

    def detect_logic(self, data):
        # attention model
        inp, att_code = self.att_coding(data)
        att_code = att_code.unsqueeze(-2)
        self.cmade.set_context(att_code)
        # self.vmade.set_context(att_code)
        # self.rmade.set_context(att_code)
        # focus model
        self.fiat.set_img(data)

        # count model
        # num_code = self.num_enc(inp)

        return att_code

    def _transform_and_evaluate(self, value, base_dist, transforms):
        # transform standard normal to target distribution and evaluate log_prob
        x = value
        log_prob = (base_dist.log_prob(x).sum(-1).unsqueeze(-1)*self.frame).sum(-2)
        for transform in transforms:
            y = transform(x)
            log_prob = log_prob - (transform.log_abs_det_jacobian(x, y).unsqueeze(-1)*self.frame).sum(-2)
            x = y

        log_prob = torch.cat((torch.zeros_like(log_prob[..., 0]).unsqueeze(-1), log_prob), -1)

        return x, log_prob

    def _z_where_scale(self, latent):
        latent[..., 1:3] = 2.0 * torch.sigmoid(latent[..., 1:3]) - 1.0
        return latent

    def num_conditional(self, latent, img_code):
        b_size = latent.size()[:-1]
        # lat_code = self.num_lat_enc(latent).unsqueeze(-2).expand(b_size + (-1,))
        # lat_code = self.num_lat_enc(latent.view(b_size[:-1] + (-1,))).unsqueeze(-2).expand(b_size + (-1,))
        img_code = img_code.expand(b_size + (-1,))
        ful_num_code = torch.cat((latent, img_code), -1)
        obj_prz  = torch.sigmoid(self.num_obj_enc(ful_num_code))
        # calculate prob for discrete distribution
        obj_abs = (1 - obj_prz).squeeze(-1)
        eos = 1e-37
        lop_sum = (torch.log(obj_prz + eos) * self.upo).sum(-2)
        cat_pro = lop_sum + torch.cat((torch.log(obj_abs+eos), torch.zeros_like(obj_abs[..., 0]).unsqueeze(-1)), -1)
        cat_pro = clamp_preserve_gradients(cat_pro, -8.0, 1.0)

        return cat_pro, torch.exp(cat_pro)
    
    def prior(self, latent):
        b_size = latent.size()[:-1]
        # prior of numbers
        log_prob_num = torch.log(self.z_num_prior)
    
        # prior of locations
        where_prior = dist.Normal(self.z_where_prior_loc.expand(b_size + (-1,)),
                                  self.z_where_prior_scale.expand(b_size + (-1,)))
        z_where = latent[..., :self.z_where_size]
        log_prob_z_where = where_prior.log_prob(z_where).sum(-1)
        log_prob_z_where = (log_prob_z_where.unsqueeze(-1) * self.upo).sum(-2)
        
        # prior of properties
        what_prior = dist.Normal(self.z_what_prior_loc.expand(b_size + (-1,)),
                                 self.z_what_prior_scale.expand(b_size + (-1,)))
        log_prob_z_what = what_prior.log_prob(latent[..., self.z_where_size:]).sum(-1)
        log_prob_z_what = (log_prob_z_what.unsqueeze(-1) * self.upo).sum(-2)
    
        # prior of occlusions (not implemented)

        log_prior = log_prob_num + log_prob_z_where + log_prob_z_what

        return log_prior

    def likelihood(self, data, latent):
        batch_size = data.size(0)
        bgm = torch.zeros(self.x_size, self.x_size, **self.options)  # black bg

        # decode individual objects
        z_where = latent[..., :self.z_where_size]
        z_watts = latent[..., self.z_where_size:]
        y_att = self.decode(z_watts)
        y = window_to_image(z_where, self.window_size, self.x_size, y_att)
        # add up to image for different number of objects
        x_mean = (y.unsqueeze(-3)*self.upo.unsqueeze(-1).unsqueeze(-1)).sum(-4)
        x_mean = x_mean + bgm

        pds = ((data - x_mean[0,:,0])**2).unsqueeze(1)
        pds = self.gaussian_smoother(pds)[0]
        pds = (pds > 0.5*torch.max(pds)).float()
        dis = ((z_where[0][...,1:].unsqueeze(-1).unsqueeze(-1) - self.cell)**2).sum(-3)
        min_dis, _ = torch.min(dis, 1)
        min_dis = (pds*min_dis).sum((-1,-2))
        rds = 100.0*(1.0-pds) + dis
        dis_min, _ = torch.min(rds.view(batch_size, self.max_obj_num, -1), -1)
        dis_min[dis_min>100.0] = 0.0
        dis_min = dis_min.sum(-1)
        reg = 5.0*dis_min + min_dis

        # define distribution and calculate log_prob
        lik_dist = dist.Normal(x_mean, self.likelihood_sd * torch.ones_like(x_mean))
        log_prob_zx = lik_dist.log_prob(data.unsqueeze(-3).unsqueeze(0)).sum((-1, -2))

        self.ims.append(x_mean[0,0,-1])

        return -reg

    def sample_all_and_prob(self, data, num_particles = 1):
        batch_size = data.size(0)
        base_dist = self.base_dist(torch.zeros(num_particles, batch_size, self.max_obj_num, self.z_prop_size, **self.options),
                                   torch.ones(num_particles, batch_size, self.max_obj_num, self.z_prop_size, **self.options))

        num_code = self.detect_logic(data)

        epsilon = base_dist.sample()
        # epsilon = self.epsilon.clone()
        # sample z_all and the assist prior
        z_all, log_prob_z = self._transform_and_evaluate(epsilon, base_dist, self.transforms)

        # sample number of objects
        log_prob_z_n, prob_z_n = self.num_conditional(z_all, num_code)
        _, inf_dex = torch.max(prob_z_n, -1)
        if self.flag == 1:
            return 0, 0, 0, inf_dex

        # sample prior
        log_prior = self.prior(z_all)

        # scale z_where
        z_all = self._z_where_scale(z_all)

        # sample likelihood
        log_prob_z_x = self.likelihood(data, z_all)

        # elbo = (prob_z_n[0]*(log_prob_z_x[0] + self.kl_weight*(log_prior[0] + self.facorrect - log_prob_z[0] - log_prob_z_n[0]))).sum((-1, -2))
        elbo = log_prob_z_x
        elbo = elbo.sum()/num_particles

        # max_lik = log_prob_z_x.gather(-1, inf_dex[0].view(-1, 1)).sum()/batch_size
        max_lik = 0.0

        self.count = self.count + 1

        return -elbo, elbo, max_lik, inf_dex


# Spatial transformer helpers.
expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])

def expand_z_where(z_where):
    out = torch.cat((torch.zeros_like(z_where[..., 0:1]), z_where), -1)
    ix = expansion_indices
    if z_where.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, -1, ix)
    out = out.view(-1, 2, 3)
    return out

# \TODO Scaling by `1/scale` here is unsatisfactory, as `scale` could be zero.
def z_where_inv(z_where):
    # [s,x,y] -> [1/s,-x/s,-y/s]
    out = torch.cat((torch.ones_like(z_where[..., 0:1]), -z_where[..., 1:]), -1)
    out = out / z_where[..., 0:1]
    return out

def window_to_image(z_where, window_size, image_size, windows):
    n = z_where.view(-1, 3).size(0)
    b_size = z_where.size()[:-1]
    assert windows.size(-1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where_inv(z_where))
    # theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(b_size + (image_size, -1))
