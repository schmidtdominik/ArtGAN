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


seed = random.randint(1, 10000)  # seed = random.randint(1, 10000) / 999
random.seed(seed)
torch.manual_seed(seed)

workers = 10  # workers for dataloader
batch_size = 128
image_size = 128
nc = 3  # Number of channels in the training images.
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 32  # Size of feature maps in generator
ndf = 32  # Size of feature maps in discriminator
num_epochs = 600
lr = 0.0002
beta1 = 0.5  # for adam
ngpu = 1
dataloader = dataset.get_dataset('celeba', image_size, batch_size, workers)

print(num_epochs)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >= 1) else "cpu")
experiment = Experiment(api_key=open('cometml.key').read().strip(), project_name="artgan3", workspace="schmidtdominik")

netG = Generator(ngpu, nz, nc, ngf).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(util.weights_init)
print(netG)

netD = Discriminator(ngpu, nz, nc, ndf).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(util.weights_init)
print(netD)

"""noise = torch.randn(batch_size, nz, 1, 1, device=device)
out = netG(noise)
out2 = netD(out)
print(out2.shape)
exit()"""

criterion = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

steps = 0

for epoch in range(num_epochs):
    experiment.log_current_epoch(epoch)
    for i, data in enumerate(dataloader, 0):
        experiment.set_step(steps)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        """
        Loss_D - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x))+log(D(G(z)))).
        Loss_G - generator loss calculated as log(D(G(z)))
        D(x) - the average output (across the batch) of the discriminator for the all real batch.
               This should start close to 1 then theoretically converge to 0.5 when G gets better. Think about why this is.
        D(G(z)) - average discriminator outputs for the all fake batch. The first number is before D is updated and the second number
                  is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better. Think about why this is.
        """

        # Output training stats
        if i % 10 == 0:
            print('step:', steps, ' epoch:', epoch)
            experiment.log_metrics({'Loss_D': errD.item(), 'Loss_G': errG.item(), 'D(x)': D_x, 'D(G(z1))': D_G_z1, 'D(G(z2))': D_G_z2})

        if (steps % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            fixed_noise = torch.randn(3, nz, 1, 1, device=device)
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu().numpy()

                plot = plt.figure(figsize=(20, 10))
                for m in range(3):
                    plt.subplot(1, 3, m + 1)
                    plt.imshow((fake[m].transpose((1, 2, 0))+1)/2)
                experiment.log_figure(figure_name='epoch_{}_{}'.format(epoch, steps))
                plt.close()

        steps += 1