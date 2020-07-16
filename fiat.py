import torch
import torch.nn as nn
import torch.nn.functional as F


# \TODO change hard-coded '3' to variables that representing an attention
class FIAT(nn.Module):

    def __init__(
            self,
            cmade,
            encode,
            win_size,
            img=torch.tensor([1.0])):
        super(FIAT, self).__init__()

        self.cmade = cmade
        self.encode = encode
        self.img = img
        self.img_size = img.size(-1)
        self.win_size = win_size
        # self.permutation = torch.cat((torch.arange(3), self.cmade.permutation + 3))

    def forward(self, x):
        z_where = x[..., :3].clone()
        z_where[..., 1:] = 2.0*torch.sigmoid(z_where[..., 1:])-1.0
        # z_where = x[..., :3]
        z_what = x[..., 3:]
        x_att = image_to_window(z_where, self.win_size, self.img_size, self.img)
        focus_code = self.encode(x_att)
        self.cmade.set_context(focus_code)
        mean, log_scale = self.cmade(z_what)
        mean = torch.cat((torch.zeros(1, 1, 1, 3).expand(mean.shape[:-1] +(-1,)).cuda(), mean), -1)
        log_scale = torch.cat((torch.zeros(1, 1, 1, 3).expand(mean.shape[:-1] +(-1,)).cuda(), log_scale), -1)
        log_scale[..., :3] = torch.Tensor([float('inf')]).cuda()
        return mean, log_scale

    def set_img(self, img):
        self.img = img
        self.img_size = img.size(-1)


# Spatial transformer helpers. Copied directly from pyro air example

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])


def expand_z_where(z_where):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    out = torch.cat((torch.zeros_like(z_where[..., 0:1]), z_where), -1)
    ix = expansion_indices
    if z_where.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, -1, ix)
    out = out.view(-1, 2, 3)
    return out


# \TODO Scaling by `1/scale` here is unsatisfactory, as `scale` could be zero.
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    out = torch.cat((torch.ones_like(z_where[..., 0:1]), -z_where[..., 1:]), -1)
    # Divide all entries by the scale.
    out = out / z_where[..., 0:1]
    return out


def image_to_window(z_where, window_size, image_size, images):
    n = z_where.view(-1, 3).size(0)
    batch_shape = z_where.size()[:-1]
    images = images.unsqueeze(-3).expand(batch_shape + (-1, -1))
    images = images.contiguous().view(-1, image_size, image_size)
    assert images.size(1) == images.size(2) == image_size, 'Size mismatch.'  # only support square images
    theta = expand_z_where(z_where)
    # theta = expand_z_where(z_where_inv(z_where))
    grid = F.affine_grid(theta, torch.Size((n, 1, window_size, window_size)))
    out = F.grid_sample(images.view(-1, 1, image_size, image_size), grid)
    return out.view(batch_shape + (-1,))
