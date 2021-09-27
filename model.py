import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator model : 

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.embed = nn.Sequential(nn.Embedding(2,50),
                                   nn.Linear(50,1))

        self.latent = nn.Sequential(nn.Linear(10,1*1*512),
                                    nn.ReLU())

        self.model = nn.Sequential(nn.ConvTranspose2d(513,512,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64,3,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.Tanh())
                                   
        def forward(self,inputs):
            noise, label = inputs
            label = self.embed(label)
            label = label.view(-1,1,1,1)
            noise = self.latent(noise)
            noise = noise.view(-1,512,1,1)
            X = torch.cat((noise,label),dim=1)
            img = self.model(X)

            return img

# Discriminator mdoel : 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.embed = nn.Sequential(nn.Embedding(2,50),
                                   nn.Linear(50,3*128*128))

        self.model = nn.Sequential(nn.Conv2d(6,64,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(4608,1),
                                   nn.Sigmoid())

        def forward(self,inputs):
            img, label = inputs
            label = self.embed(label)
            label = label.view(-1,3,128,128)
            X = torch.cat((img,label),dim=1)
            output = self.model(X)

            return output

# Initialising:

discriminator = Discriminator()
generator = Generator()
discriminator.to(device)
generator.to(device)

print(generator)