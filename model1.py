import torch
import torch.nn as nn

# Generator:
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        # Label should be embedded and changed to (128,1,8,8) in order to concatenate with noise of same input
        self.embed = nn.Embedding(2,8*8)

        # Noise of input dim (128,100) is changed to (128,511,8,8)
        self.changenoise = nn.Sequential(nn.Linear(100,511*64), nn.ReLU())

        self.model = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128,64,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64,32,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(32,16,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(16,8,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(8,1,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.Tanh())

    def forward(self,noise,label):
        label = self.embed(label)
        label = label.view(-1,1,8,8)
        noise = self.changenoise(noise)
        noise = noise.view(-1,511,8,8)
        input = torch.cat((noise,label),dim=1)
        image = self.model(input)
        return image

# Discriminator:
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        # Label should be embedded and changed to (128,1,8,8) in order to concatenate with image of same input
        self.embed = nn.Embedding(2,8*8)

        self.model = nn.Sequential(nn.Conv2d(2,32,kernel_size=3,stride=2,bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),   # output - (128,32,4,4)
                                   nn.Conv2d(32,64,kernel_size=3,stride=2,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),   # output - (128,64,2,2)
                                   nn.Flatten(),
                                   nn.Linear(64,1),
                                   nn.Sigmoid())

    def forward(self,image,label):
        label = self.embed(label)
        label = label.view(-1,1,8,8)
        input = torch.cat((image,label),dim=1)
        output = self.model(input)
        return output

generator = Generator()
discriminator = Discriminator()