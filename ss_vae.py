import argparse

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from utils.dataset_cached import setup_data_loaders, CELEBA_EASY_LABELS
from utils.dataset_cached import CELEBACached
from models.ccvae import CCVAE

import numpy as np
import os


def main(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """

    im_shape = (3, 64, 64)

    data_loaders = setup_data_loaders(args.cuda,
                                      args.batch_size,
                                      cache_data=True,
                                      sup_frac=args.sup_frac,
                                      root='./data/datasets/celeba')


    cc_vae = CCVAE(z_dim=args.z_dim,
                   num_classes=len(CELEBA_EASY_LABELS),
                   im_shape=im_shape,
                   use_cuda=args.cuda,
                   prior_fn=data_loaders['test'].dataset.prior_fn)

    optim = torch.optim.Adam(params=cc_vae.parameters(), lr=args.learning_rate)

    # run inference for a certain number of epochs
    for epoch in range(0, args.num_epochs):

        # # # compute number of batches for an epoch
        if args.sup_frac == 1.0: # fullt supervised
            batches_per_epoch = len(data_loaders["sup"])
            period_sup_batches = 1
            sup_batches = batches_per_epoch
        elif args.sup_frac > 0.0: # semi-supervised
            sup_batches = len(data_loaders["sup"])
            unsup_batches = len(data_loaders["unsup"])
            batches_per_epoch = sup_batches + unsup_batches
            period_sup_batches = int(batches_per_epoch / sup_batches)
        elif args.sup_frac == 0.0: # unsupervised
            sup_batches = 0.0
            batches_per_epoch = len(data_loaders["unsup"])
            period_sup_batches = np.Inf
        else:
            assert False, "Data frac not correct"

        # initialize variables to store loss values
        epoch_losses_sup = 0.0
        epoch_losses_unsup = 0.0

        # setup the iterators for training data loaders
        if args.sup_frac != 0.0:
            sup_iter = iter(data_loaders["sup"])
        if args.sup_frac != 1.0:
            unsup_iter = iter(data_loaders["unsup"])

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0

        for i in tqdm(range(batches_per_epoch)):
            # whether this batch is supervised or not
            is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches
            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)
            
            if args.cuda:
                xs, ys = xs.cuda(), ys.cuda()

            if is_supervised:
                loss = cc_vae.sup(xs, ys)
                epoch_losses_sup += loss.detach().item()
            else:
                loss = cc_vae.unsup(xs)
                epoch_losses_unsup += loss.detach().item()

            loss.backward()
            optim.step()
            optim.zero_grad()
            
        if args.sup_frac != 0.0:        
            with torch.no_grad():
                validation_accuracy = cc_vae.accuracy(data_loaders['valid'])
        else:
            validation_accuracy = np.nan

        with torch.no_grad():
            # save some reconstructions
            img = CELEBACached.fixed_imgs
            if args.cuda:
                img = img.cuda()
            recon = cc_vae.reconstruct_img(img).view(-1, *im_shape)
            save_image(make_grid(recon, nrow=8), './data/output/recon.png')
            save_image(make_grid(img, nrow=8), './data/output/img.png')
        
        print("[Epoch %03d] Sup Loss %.3f, Unsup Loss %.3f, Val Acc %.3f" % 
                (epoch, epoch_losses_sup, epoch_losses_unsup, validation_accuracy))
    cc_vae.save_models(args.data_dir)
    test_acc = cc_vae.accuracy(data_loaders['test'])
    print("Test acc %.3f" % test_acc)
    cc_vae.latent_walk(img[5], './data/output')
    return 

def parser_args(parser):
    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=200, type=int,
                        help="number of epochs to run")
    parser.add_argument('-sup', '--sup-frac', default=1.0,
                        type=float, help="supervised fractional amount of the data i.e. "
                                         "how many of the images have supervised labels."
                                         "Should be a multiple of train_size / batch_size")
    parser.add_argument('-zd', '--z_dim', default=45, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data path')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    main(args)

