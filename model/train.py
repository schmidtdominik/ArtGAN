import itertools

from comet_ml import Experiment

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

import torchvision.utils as vutils
from matplotlib import pyplot as plt
import numpy as np

import dataset, util
from gen_disc import Generator, Discriminator
from torch import autograd
from tqdm import tqdm

seed = random.randint(1, 10000)  # seed = random.randint(1, 10000) / 999
random.seed(seed)
torch.manual_seed(seed)

workers = 10  # workers for dataloader
batch_size = 64
image_size = 128
nc = 3  # Number of channels in the training images.
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64   # Size of feature maps in discriminator
num_epochs = 600
ngpu = 1
n_critic = 5
lambda_ = 10
dataloader = dataset.get_dataset('art', image_size, batch_size, workers)

print(num_epochs)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >= 1) else "cpu")
experiment = Experiment(api_key=open('cometml.key').read().strip(), project_name="artgan3", workspace="schmidtdominik")

netG = Generator(ngpu, nz, nc, ngf).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
#netG.apply(util.weights_init)

netD = Discriminator(ngpu, nz, nc, ndf).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
#netD.apply(util.weights_init)

# Establish convention for real and fake labels during training
one_label = 1
mone_label = -1

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(-1, real_data.shape[1], real_data.shape[2], real_data.shape[3])
    #alpha = alpha.expand(batch_size, real_data.nelement()/batch_size).contiguous().view(batch_size, 3, 32, 32)
    alpha = alpha.to(device) if (ngpu >= 1) else alpha

    #  The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 2
    #print('a', alpha.shape)
    #print('b', real_data.shape)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if (ngpu >= 1):
        interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if (ngpu >= 1) else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty

for epoch in range(num_epochs):
    inf_dl = get_infinite_batches(dataloader)
    minus_one = torch.as_tensor(-1, dtype=torch.float).to(device)
    one = torch.as_tensor(1, dtype=torch.float).to(device)
    for i in tqdm(itertools.count()):
        experiment.set_step(i)

        for p in netD.parameters():
            p.requires_grad_(True)

        for t in range(n_critic):
            optimizerD.zero_grad()
            netD.zero_grad()

            images = inf_dl.__next__()
            images_tra = images.to(device)
            b_size = images_tra.size(0)
            if (images.size()[0] != batch_size):
                continue

            D_real = netD(images_tra)
            D_real = D_real.mean()
            D_real.backward(minus_one)

            # train with fake
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise).detach()
            D_fake = netD(fake).mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, images_tra, fake)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # G update

        for p in netD.parameters():
            p.requires_grad_(False)

        netG.zero_grad()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        G_loss = netD(fake).mean()
        G_loss.backward(minus_one)
        G_cost = -G_loss
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            print('step:', i)

            stats = {'D_cost': D_cost.item(), 'G_cost': G_cost.item(), 'Wasserstein_D': Wasserstein_D.item(), 'D_real': D_real.item(), 'D_fake': D_fake.item()}
            print(stats)
            experiment.log_metrics(stats)

        if (i % 80 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            fixed_noise = torch.randn(3, nz, 1, 1, device=device)
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu().numpy()

                plot = plt.figure(figsize=(20, 10))
                for m in range(3):
                    plt.subplot(1, 3, m + 1)
                    plt.imshow((fake[m].transpose((1, 2, 0))+1)/2)
                experiment.log_figure(figure_name='step_{}'.format(i))
                plt.close()