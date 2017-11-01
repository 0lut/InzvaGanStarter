import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
from utils import ChunkSampler
class dataSets():
    def __init__(self, batch_size=128, num_train=49000, num_val=1000):
        transforms_normal = T.Compose([T.ToTensor()])
        self.mnist_training = dset.MNIST('./datasets/MNIST', train=True, transform=transforms_normal,download=True)
        self.mnist_train_loader = DataLoader(self.mnist_training, batch_size=batch_size, sampler=ChunkSampler(num_train, 0))
       

