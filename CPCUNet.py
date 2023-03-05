import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from BaseModel import BaseModel
from torchvision.transforms.functional import resize

class UNetRes50Enc(nn.Module):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
        super(UNetRes50Enc, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate

        resnet50 = torchvision.models.resnet50(weights="DEFAULT")
        self.e1 = nn.Sequential(resnet50._modules['conv1'], resnet50._modules['bn1'], resnet50._modules['relu'])
        self.e2 = nn.Sequential(resnet50._modules['maxpool'], resnet50._modules['layer1'])
        self.e3 = nn.Sequential(resnet50._modules['layer2'])
        self.e4 = nn.Sequential(resnet50._modules['layer3'])
        
        for param in self.e1.parameters():
            param.requires_grad_(False)
            
        for param in self.e2.parameters():
            param.requires_grad_(False)
            
        for param in self.e3.parameters():
            param.requires_grad_(False)
            
        for param in self.e4.parameters():
            param.requires_grad_(False)

        if (loss_fn is None):
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = loss_fn

        if (optimizer is None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)


        self = self.to(self.device)


    def forward(self, X):
        X = X.to(self.device)

        s1 = self.e1(X)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)

        return (X, s1, s2, s3, s4)

class UNetRes50(BaseModel):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None, aux=None):
        super(UNetRes50, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate
        
        if (aux is None):
            self.aux = 0
        else:
            self.aux = aux
        
        
        self.c1 = self.convblock(1024+self.aux, 512)
        self.c2 = self.convblock(512+self.aux, 256)
        self.c3 = self.convblock(192+self.aux, 128)
        self.c4 = self.convblock(67+self.aux, 64)
        
        self.u1 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2))
        self.u2 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.u3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.u4 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        
        self.out = nn.Conv2d(64+self.aux, 1, (1, 1), padding="same")

        if (loss_fn is None):
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = loss_fn

        if (optimizer is None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)


        self = self.to(self.device)

    def convblock(self, in_features, out_features):
        block = nn.Sequential(
            nn.Conv2d(in_features, out_features, (3, 3), padding="same"),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, (3, 3), padding="same"),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
            )
        block = block.to(self.device)
        return block


    def forward(self, X0, Xaux):
        
        X, s1, s2, s3, s4 = X0
        
        
        X, s1, s2, s3, s4 = X.to(self.device), s1.to(self.device), s2.to(self.device), s3.to(self.device), s4.to(self.device)
        Xaux = Xaux.to(self.device)

        s1 = self.e1(X)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        
        s4 = self.u1(s4)
        s4 = torch.cat([resize(Xaux, (s4.shape[2], s4.shape[3])), s4], dim=1)
        s3 = self.c1(torch.cat([s3, s4], dim=1))

        s3 = self.u2(s3)
        s3 = torch.cat([resize(Xaux, (s3.shape[2], s3.shape[3])), s3], dim=1)
        s2 = self.c2(torch.cat([s2, s3], dim=1))

        s2 = self.u3(s2)
        s2 = torch.cat([resize(Xaux, (s2.shape[2], s2.shape[3])), s2], dim=1)
        s1 = self.c3(torch.cat([s1, s2], dim=1))

        s1 = self.u4(s1)
        s1 = torch.cat([resize(Xaux, (s1.shape[2], s1.shape[3])), s1], dim=1)
        X = self.c4(torch.cat([X, s1], dim=1))

        X = torch.cat([resize(Xaux, (X.shape[2], X.shape[3])), X], dim=1)
        X = torch.sigmoid(self.out(X))

        return X
