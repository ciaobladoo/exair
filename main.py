"""
AIR applied to the multi-mnist data set [1]. Exchangeable Version

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""
from __future__ import division

import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim

import pyro.contrib.examples.multi_mnist as multi_mnist
from pyro.contrib.examples.util import get_data_directory

from air_exchange import EAIR
from modules import FCNets, GaussianSmoothing


def count_accuracy(X, true_counts, air, batch_size):
    assert X.size(0) == true_counts.size(0), 'Size mismatch.'
    assert X.size(0) % batch_size == 0, 'Input size must be multiple of batch_size.'
    correct_counts = 0.0
    total_counts = 0.0

    for i in range(X.size(0) // batch_size):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        true_counts_batch = true_counts[i * batch_size:(i + 1) * batch_size]
        _, _, _, inferred_counts = air.sample_all_and_prob(X_batch)
        total_counts += inferred_counts[0].sum()
        correct_ind = (true_counts_batch == inferred_counts[0].float())
        correct_counts += correct_ind.sum()

    acc = correct_counts.float()/ X.size(0)
    return acc, total_counts


def make_prior(k):
    # implementation for N=3 only
    assert 0 < k <= 1
    u = 1 / (1 + k + k ** 2 + k ** 3)
    p0 = 1 - u
    p1 = 1 - (k * u) / p0
    p2 = 1 - (k ** 2 * u) / (p0 * p1)
    cat_dist = [1 - p0, p0 * (1 - p1), p0 * p1 * (1 - p2), p0 * p1 * p2]
    cat_dist = torch.tensor(np.array(cat_dist, dtype=np.float32))
    return cat_dist


def truncated_geometric_prior(k, num):
    assert 0 < k <= 1
    idx = torch.arange(0, num+1).float()
    up = torch.exp(idx*torch.log(torch.tensor(1-k)) + torch.log(torch.tensor(k)))
    down = 1-(1-k)**(num+1)
    return up/down


def load_data():
    inpath = get_data_directory(__file__)
    X_np, Y = multi_mnist.load(inpath)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = torch.from_numpy(X_np)
    # Using FloatTensor to allow comparison with values sampled from Bernoulli.
    counts = torch.FloatTensor([len(objs) for objs in Y])
    return X, counts


def main(**kwargs):

    args = argparse.Namespace(**kwargs)

    if 'save' in args:
        if os.path.exists(args.save):
            raise RuntimeError('Output file "{}" already exists.'.format(args.save))

    X, true_counts = load_data()
    X = X[8].unsqueeze(0)
    X_size = X.size(0)
    true_counts = torch.ones(1)
    if args.cuda:
        X = X.cuda()
        true_counts = true_counts.cuda()

    model_arg_keys = ['window_size',
                      'decoder_output_bias',
                      'decoder_output_use_sigmoid',
                      'encoder_net',
                      'decoder_net',
                      'foc_net',
                      'att_net',
                      'num_lat_net',
                      'num_obj_net',
                      'cmade',
                      'fmade',
                      'z_pres_prior',
                      'z_where_prior_loc',
                      'z_where_prior_scale',
                      'z_what_prior_loc',
                      'z_what_prior_scale',
                      'likelihood_sd']
    model_args = {key: getattr(args, key) for key in model_arg_keys if key in args}
    air = EAIR(
        max_obj_num=args.max_obj_num,
        x_size=50,
        z_what_size=args.encoder_latent_size,
        z_num_func=truncated_geometric_prior,
        use_cuda=args.cuda,
        **model_args
    )

    if args.verbose:
        print(air)
        print(args)

    if 'load' in args:
        print('Loading parameters...')
        air.load_state_dict(torch.load(args.load))

    adam = optim.Adam(air.parameters(), lr = args.learning_rate)

    # Do inference.
    t0 = time.time()
    # examples_to_viz = X[5:10]

    # acc log
    count_acc = []
    likes = []
    elbow = []

    # torch.autograd.set_detect_anomaly(True)
    for i in range(1, args.num_steps + 1):

        adam.zero_grad()

        # idx = torch.multinomial(torch.ones(X_size), args.batch_size, replacement=False).to(X.device)
        (loss, elbo, ike, inf_dex) = air.sample_all_and_prob(X, args.num_particles)

        if torch.isnan(elbo):
            torch.save(air.state_dict(), 'fuck.pt')

        loss.backward()
        adam.step()

        if args.progress_every > 0 and i % args.progress_every == 0:
            print('i={}, epochs={:.2f}, elapsed={:.2f}, elbo={:.2f}'.format(
                i,
                (i * args.batch_size) / X_size,
                (time.time() - t0) / 3600,
                -loss/args.batch_size))

        if args.viz and i % args.viz_every == 0:
            pass

        if args.eval_every > 0 and i % args.eval_every == 0:
            elbow.append(elbo/args.batch_size)
            likes.append(ike)
            # Measure accuracy on subset of training data.
            torch.cuda.empty_cache()
            air.flag = 1
            with torch.no_grad():
                acc, total_counts = count_accuracy(X, true_counts, air, 1000)
            count_acc.append(acc)
            air.flag = 0
            print('i={}, accuracy={}, total_counts={}'.format(i, acc, total_counts))

        if 'save' in args and i % args.save_every == 0:
            print('Saving parameters...')
            torch.save(air.state_dict(), args.save)

    elbow = torch.tensor(elbow)
    torch.save(elbow, 'elbo.pt')
    likes = torch.tensor(likes)
    torch.save(likes, 'like.pt')
    acc = torch.tensor(count_acc)
    torch.save(acc, 'acc.pt')
    ims = torch.stack(air.ims)
    torch.save(ims, 'ims.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pyro AIR example", argument_default=argparse.SUPPRESS)
    parser.add_argument('-n', '--num-steps', type=int, default=int(1e8),
                        help='number of optimization steps to take')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-p', '--num-particles', type=int, default=1,
                        help='number of partiles used to estimate elbo')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--progress-every', type=int, default=1,
                        help='number of steps between writing progress to stdout')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='number of steps between evaluations')
    parser.add_argument('--decoder-net', type=int, nargs='+', default=[200],
                        help='decoder net hidden layer sizes')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='generate vizualizations during optimization')
    parser.add_argument('--viz-every', type=int, default=100,
                        help='number of steps between vizualizations')
    parser.add_argument('--visdom-env', default='main',
                        help='visdom enviroment name')
    parser.add_argument('--load', type=str,
                        help='load previously saved parameters')
    parser.add_argument('--save', type=str,
                        help='save parameters to specified file')
    parser.add_argument('--save-every', type=int, default=1e4,
                        help='number of steps between parameter saves')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='use PyTorch jit')
    parser.add_argument('-t', '--max-obj-num', type=int, default=3,
                        help='maximum number of objects')
    parser.add_argument('--rnn-hidden-size', type=int, nargs='+', default=256,
                        help='attention net hidden layers')
    parser.add_argument('--foc-net', type=int, nargs='+', default=[200],
                        help='focus net hidden layers')
    parser.add_argument('--att-net', type=int, nargs='+', default=[256],
                        help='attention net hidden layers')
    parser.add_argument('--num-lat-net', type=tuple, default=([128], [64]),
                        help='number latent net hidden layers')
    parser.add_argument('--num-obj-net', type=int, nargs='+', default=[200],
                        help='number object net hidden layers')
    parser.add_argument('--cmade', type=int, nargs="+", default=[200],
                        help='cmade hidden layers')
    parser.add_argument('--fmade', type=int, nargs="+", default=[],
                        help='fmade hidden layers')
    parser.add_argument('--encoder-latent-size', type=int, default=50,
                        help='attention window encoder/decoder latent space size')
    parser.add_argument('--decoder-output-bias', type=float,
                        help='bias added to decoder output (prior to applying non-linearity)')
    parser.add_argument('--decoder-output-use-sigmoid', action='store_true',
                        help='apply sigmoid function to output of decoder network')
    parser.add_argument('--window-size', type=int, default=28,
                        help='attention window size')
    parser.add_argument('--z-pres-prior', type=float, default=0.5,
                        help='prior success probability for z_pres')
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='write hyper parameters and network architecture to stdout')
    main(**vars(parser.parse_args()))
