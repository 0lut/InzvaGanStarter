from utils import *
class Generator(nn.Module):
    def __init__(self, NOISE_DIM, dtype):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(NOISE_DIM,1024),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(1024),
                            nn.Linear(1024, 7*7*128),
                            nn.BatchNorm1d(7*7*128),
                            Unflatten(-1, 128, 7, 7),
                            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                            nn.Tanh(),
                            Flatten())
                            
                            
        self.model.type(dtype)
    def forward(self, input):
        out = self.model(input)
        return out

class Discriminator(nn.Module):
    def __init__(self, batch_size, dtype):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, kernel_size=5, stride=1),
        nn.LeakyReLU(1e-2),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.LeakyReLU(1e-2),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(1e-2),
        nn.Linear(1024, 1)
    )

        self.model.type(dtype)
    def forward(self, input):
        out = self.model(input)
        return out


# Calculates the following:
# bce(s,y) = y * log(s) + (1-y) * log(1-s)
# I dont know PyTorch has implementation for this, so used written by CS231n instructors. (cs231n.github.io)
def bce_loss(input, target):

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

# Note that these are negated from the equations presented earlier as we will be minimizing these losses therefore there is no need to care for negating the results.
# Discriminator loss =  - Expectation over [ log ( D(x) ) + (1 - log (D (G(z) ) ) ) ]
def discriminator_loss(logits_real, logits_fake):
    '''
    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    '''

    return loss
        
# Generator loss = - Expectation over the (log (D (G(z) ) ) )

def generator_loss(logits_fake):
    '''    
    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    '''
    return loss



def get_optimizer(model):
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer
