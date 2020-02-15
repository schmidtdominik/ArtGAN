import itertools
from time import time

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

workers = 16
ngpu = 1

batch_size = 64
num_epochs = 600

n_critic = 5
lambda_ = 10

depth_modifier = 16
# (DEPTH):    512 256 128 64 32 16
resolution_level_count = 6 # 4x4 is lowest, 2**(resolution_levels+1) is highest
resolution_levels = [int(k) for k in 2**(2+np.arange(resolution_level_count))]
depth_levels = [32, 16, 8, 4, 2, 1] # insert 32 on the left
total_training_minutes = 10  # in minutes
total_training_time = total_training_minutes*60
training_time_per_level = 1.9**np.arange(resolution_level_count) #1.85 is a training hyperparameter
training_time_per_level = (training_time_per_level/np.sum(training_time_per_level))*total_training_time
assert len(depth_levels) == len(resolution_levels) == len(training_time_per_level)
image_size = resolution_levels[-1]
z_size = depth_levels[0]*depth_modifier
print('output image size:', image_size)
print('z size:', z_size)

dataset_str = 'celeba'

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >= 1) else "cpu")
experiment = Experiment(api_key=open('cometml.key').read().strip(), project_name="artgan3", workspace="schmidtdominik")

netG = Generator(z_size, depth_modifier, resolution_levels, depth_levels).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
#netG.apply(util.weights_init)

netD = Discriminator(z_size, depth_modifier, resolution_levels, depth_levels).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
#netD.apply(util.weights_init)

# Establish convention for real and fake labels during training

"""
# progressive growing tests

netD.current_level = 5
netD.transition_phase = True
alpha = torch.as_tensor(0, dtype=torch.float, device=device)
netD.alpha = alpha

dataloader = dataset.get_dataset(dataset_str, resolution_levels[netD.current_level], batch_size, workers)
data_cycler = dataset.get_infinite_batches(dataloader)

images = data_cycler.__next__()
images_tra = images.to(device)
b_size = images_tra.size(0)

out = netD(images_tra)
print(out.shape)

exit()"""


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))



def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(-1, real_data.shape[1], real_data.shape[2], real_data.shape[3])
    #alpha = alpha.expand(batch_size, real_data.nelement()/batch_size).contiguous().view(batch_size, 3, 32, 32)
    alpha = alpha.to(device) if (ngpu >= 1) else alpha

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


transition_phase = False
current_level = 5

minus_one = torch.as_tensor(-1, dtype=torch.float, device=device)
one = torch.as_tensor(1, dtype=torch.float, device=device)
dataloader = dataset.get_dataset(dataset_str, resolution_levels[current_level], batch_size, workers)
data_cycler = dataset.get_infinite_batches(dataloader)
level_begin = time()
for i in tqdm(itertools.count()):
    experiment.set_step(i)
    if time()-level_begin > training_time_per_level[current_level]/2:
        if transition_phase:
            level_begin = time()
            transition_phase = False
        elif current_level != resolution_level_count-1:
            level_begin = time()
            current_level += 1
            dataloader = dataset.get_dataset(dataset_str, resolution_levels[current_level], batch_size, workers)
            data_cycler = dataset.get_infinite_batches(dataloader)
            transition_phase = True
    assert not (transition_phase and current_level == 0)
    netG.current_level = current_level
    netD.current_level = current_level
    netG.transition_phase = transition_phase
    netD.transition_phase = transition_phase
    growing_alpha = torch.as_tensor((time() - level_begin)/(training_time_per_level[current_level]/2) if transition_phase else -2, dtype=torch.float, device=device)
    netG.alpha = growing_alpha
    netD.alpha = growing_alpha
    print(current_level, transition_phase, growing_alpha.item())

    for p in netD.parameters():
        p.requires_grad_(True)

    for t in range(n_critic):
        optimizerD.zero_grad()
        netD.zero_grad()

        images = data_cycler.__next__()
        images_tra = images.to(device)
        b_size = images_tra.size(0)
        if (images.size()[0] != batch_size):
            continue

        D_real = netD(images_tra)
        D_real = D_real.mean()
        D_real.backward(minus_one)

        # train with fake
        noise = torch.randn(b_size, z_size, 1, 1, device=device)
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

    noise = torch.randn(batch_size, z_size, 1, 1, device=device)
    fake = netG(noise)
    G_loss = netD(fake).mean()
    G_loss.backward(minus_one)
    G_cost = -G_loss
    optimizerG.step()

    # Output training stats
    if i % 10 == 0:
        print('step:', i)

        stats = {'D_cost': D_cost.item(), 'G_cost': G_cost.item(), 'Wasserstein_D': Wasserstein_D.item(), 'D_real': D_real.item(), 'D_fake': D_fake.item(), 'transition_phase': int(transition_phase), 'alpha': growing_alpha.item(), 'level_begin': level_begin}
        print(stats)
        experiment.log_metrics(stats)

    if (i % 80 == 0):
        fixed_noise = torch.randn(3, z_size, 1, 1, device=device)
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu().numpy()

            plot = plt.figure(figsize=(20, 10))
            for m in range(3):
                plt.subplot(1, 3, m + 1)
                plt.imshow((fake[m].transpose((1, 2, 0))+1)/2)
            experiment.log_figure(figure_name='step_{}'.format(i))
            plt.close()