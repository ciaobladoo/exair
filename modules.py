import torch
import torch.nn as nn
from torch.nn.functional import softplus
import torchvision.models.resnet as resnet
import math
import numbers
from torch.nn import functional as F


# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
class Encoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, non_linear_layer):
        super(Encoder, self).__init__()
        self.z_size = z_size
        output_size = 2 * z_size
        self.mlp = MLP(x_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, x):
        a = self.mlp(x)
        return a[:, 0:self.z_size], softplus(a[:, self.z_size:])


# Takes a latent code, z_what, to pixel intensities.
class Decoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, bias, use_sigmoid, non_linear_layer):
        super(Decoder, self).__init__()
        self.bias = bias
        self.use_sigmoid = use_sigmoid
        self.mlp = MLP(z_size, h_sizes + [x_size], non_linear_layer)

    def forward(self, z):
        a = self.mlp(z)
        if self.bias is not None:
            a = a + self.bias
        return torch.sigmoid(a) if self.use_sigmoid else a


# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(self, in_size, out_sizes, non_linear_layer, output_non_linearity=False):
        super(MLP, self).__init__()
        assert len(out_sizes) >= 1
        layers = []
        in_sizes = [in_size] + out_sizes[0:-1]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)
        self.output_size = out_sizes[-1]

    def forward(self, x):
        return self.seq(x)

# permutation invariant linear network (sum-decomposable network)
class SDN(nn.Module):
    def __init__(self, pool, in_size, out_sizes1, nl1, out_sizes2, nl2, output_non_linearity=False):
        super(SDN, self).__init__()
        assert len(out_sizes1) >=1 and len(out_sizes2) >=1
        self.pool = pool
        layers_in = []
        in_sizes = [in_size] + out_sizes1[0:-1]
        sizes = list(zip(in_sizes, out_sizes1))
        for (i, o) in sizes[0:-1]:
            layers_in.append(nn.Linear(i, o))
            layers_in.append(nl1())
        layers_in.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        layers_in.append(nl1())

        layers_out = []
        in_sizes = [out_sizes1[-1]] + out_sizes2[0:-1]
        sizes = list(zip(in_sizes, out_sizes2))
        for (i, o) in sizes[0:-1]:
            layers_out.append(nn.Linear(i, o))
            layers_out.append(nl2())
        layers_out.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers_out.append(nl2())
        self.seq_in = nn.Sequential(*layers_in)
        self.seq_out = nn.Sequential(*layers_out)
        self.output_size = out_sizes2[-1]

    def forward(self, x):
        hidden = self.seq_in(x)
        pool = self.pool(hidden, -2)
        return self.seq_out(pool)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SingleChannelResNet(resnet.ResNet):
    def __init__(self, n_channels, block, layers, num_cls = 10):
        super(SingleChannelResNet, self).__init__(block, layers, num_classes=num_cls)
        self.conv1 = nn.Conv2d(n_channels, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)
        self.avgpool = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=4, stride=4)

    def forward(self, x):
        return super(SingleChannelResNet, self).forward(x)


class ConvNets(nn.Module):
    def __init__(self, n_channel, att_net, nl, out_size):
        super(ConvNets, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, att_net, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(att_net, att_net, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(att_net, att_net, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(att_net, att_net, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(att_net, att_net, kernel_size=4, stride=3)
        self.conv7 = nn.Conv2d(att_net, att_net, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(att_net)
        self.bn2 = nn.BatchNorm2d(att_net)
        self.bn3 = nn.BatchNorm2d(att_net)
        self.bn4 = nn.BatchNorm2d(att_net)
        self.fc = nn.Linear(att_net, out_size)
        self.nl = nl()

    def forward(self, x):
        x = self.conv1(x)
        x = self.nl(x)
        x = self.conv2(x)
        x = self.nl(x)
        x = self.conv3(x)
        x = self.nl(x)
        x = self.conv4(x)
        x = self.nl(x)
        x = self.conv5(x)
        x = self.nl(x)
        while x.size(-1) != 1:
            x = self.conv7(x)
            x = self.nl(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class FCNets(nn.Module):
    def __init__(self, n_channel, num_classes=2):
        super(FCNets, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = nn.Sigmoid()(x)

        return x


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, stride, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        # self.padding = [int((size-1)/2) for size in kernel_size]
        self.stride = stride
        self.padding = 0

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding = self.padding, stride = self.stride)


class Blob2Point(nn.Module):
    """apply two conv-net iteratively to shrink blobs to points individually
    the input needs to be one channel and 2d
    Argument:
        circle_size: the kernel size of the conv-net used to mark object center
    """
    def __init__(self):
        super(Blob2Point, self).__init__()

        w1 = torch.zeros(9,1,3,3)
        for i in range(9):
            w1[i,0,math.floor(i/3), i%3] = 1.0
        w1 = torch.cat((w1[:4], w1[5:]), 0)
        w1[:,0,1,1] = -1.0
        self.register_buffer('w1', w1)
        w2 = torch.ones(1,1,7,7)
        self.register_buffer('w2', w2)

        self.conv = F.conv2d

    def one_step(self, x):
        y = (x > 0).float() * (
                (F.conv2d(x, weight=self.w1, padding=[1, 1]) != 0.0).float().sum(1, keepdim=True) == 0.0).float()
        x = x + y
        return x

    def forward(self, x):
        # for i in range(5):
        #     x = self.one_step(x)
        x = F.conv2d(x, weight=self.w2, padding=[3, 3])
        x = (x > 0).float() * (
                1.0 - ((F.conv2d(x, weight=self.w1, padding=[1, 1]) > 0.0).float().sum(1, keepdim=True) > 0.0).float())

        return x
