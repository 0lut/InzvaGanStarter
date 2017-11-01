from loadDset import dataSets
from models_losses import Generator, Discriminator, get_optimizer
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import sample_noise

NUM_TRAIN = 49000
batch_size = 128
NOISE_DIM = 96
img_size = 28*28
num_epochs = 10
data = dataSets()
dtype = torch.cuda.FloatTensor 
#if you are running it on CPU, then torch.FloatTensor

G = Generator(NOISE_DIM,dtype)
D = Discriminator(batch_size, dtype)

G_optim = get_optimizer(G)
D_optim = get_optimizer(D)


#imgs = data.mnist_train_loader.__iter__().next()[0].view(batch_size,1,img_size).numpy().squeeze()




for epoch in range(num_epochs):
    for x, _ in data.mnist_train_loader:
        if len(x) != batch_size:
            continue

    
        D_optim.zero_grad()
        real_images = Variable(x).type(dtype)
        d_random_noise = Variable(sample_noise(batch_size, NOISE_DIM)).type(dtype)
        #Discriminator loss
        #Feed generator with random_noise and create fake image
        #Then feed discriminator two times, once with fake images, once with real ones
        #Calculate loss and do a backward pass.
        #dont forget to .detach generated images to prevent gradient calculation over them.


        G_optim.zero_grad()

        g_random_noise = Variable(sample_noise(batch_size, NOISE_DIM)).type(dtype)
        #Generator loss
        #Create fake images with generator
        #Feed your discriminator with new synthesised images
        #Calculate loss and do a backward pass.
