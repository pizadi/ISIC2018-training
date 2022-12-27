import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from BaseModel import BaseModel

class SeqConvLSTM2d(nn.Module):
    def __init__(self, in_features, hidden_features, kernel_size, device=None):
        super(SeqConvLSTM2d, self).__init__()
        
        self.hidden_features = hidden_features

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


        self.conv_i = nn.Conv2d(in_features+hidden_features, hidden_features, kernel_size, padding="same")
        self.conv_f = nn.Conv2d(in_features+hidden_features, hidden_features, kernel_size, padding="same")
        self.conv_g = nn.Conv2d(in_features+hidden_features, hidden_features, kernel_size, padding="same")
        self.conv_o = nn.Conv2d(in_features+hidden_features, hidden_features, kernel_size, padding="same")

        self = self.to(self.device)

    def forward(self, X):
        h_shape = [X.shape[0], self.hidden_features, *X.shape[3:]]
        h, c = torch.randn(h_shape)/X.shape[2], torch.randn(h_shape)/X.shape[2]
        X, h, c = X.to(self.device), h.to(self.device), c.to(self.device)

        for t in range(X.shape[1]):
            intensor = X[:,t,:,:,:]
            intensor = torch.cat([intensor, h], dim=1)
            i = torch.sigmoid(self.conv_i(intensor))
            f = torch.sigmoid(self.conv_f(intensor))
            g = torch.tanh(self.conv_g(intensor))
            o = torch.sigmoid(self.conv_o(intensor))
            c = f*c + i*g
            h = o*torch.tanh(c)

        for t in range(X.shape[1]-2, -1, -1):
            intensor = X[:,t,:,:,:]
            intensor = torch.cat([intensor, h], dim=1)
            i = torch.sigmoid(self.conv_i(intensor))
            f = torch.sigmoid(self.conv_f(intensor))
            g = torch.tanh(self.conv_g(intensor))
            o = torch.sigmoid(self.conv_o(intensor))
            c = f*c + i*g
            h = o*torch.tanh(c)

        return h

class BCDUnetD3(BaseModel):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
        super(BCDUnetD3, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate

        self.block1 = nn.Sequential(    
            nn.Conv2d(3, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.ReLU())

        self.block2 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding="same"),
            nn.ReLU())

        self.block3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding="same"),
            nn.ReLU())
        
        self.drop3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        #D1
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.5))

        #D2
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.5))

        #D3
        self.block6 = nn.Sequential(
            nn.Conv2d(1024, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.block7 = nn.Sequential(
            SeqConvLSTM2d(256, 128, (3, 3)),
            nn.Conv2d(128, 256, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding="same"),
            nn.ReLU())

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.block8 = nn.Sequential(
            SeqConvLSTM2d(128, 64, (3, 3)),
            nn.Conv2d(64, 128, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding="same"),
            nn.ReLU())

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.block9 = nn.Sequential(
            SeqConvLSTM2d(64, 32, (3, 3)),
            nn.Conv2d(32, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 2, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(2, 1, (1, 1), padding="same"),
            nn.Sigmoid())

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

        s1 = self.block1(X)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.pool3(self.block4(s3))
        b = self.block5(s4)
        b = torch.cat([b, s4], dim=1)
        b = self.block6(b)
        b = self.up7(b)
        b = b[:,None,:,:,:]
        d3 = self.drop3(s3)[:,None,:,:,:]
        b = torch.cat([b, d3], dim=1)
        b = self.block7(b)
        b = self.up8(b)
        b = b[:,None,:,:,:]
        s2 = s2[:,None,:,:,:]
        b = torch.cat([b, s2], dim=1)
        b = self.block8(b)
        b = self.up9(b)
        b = b[:,None,:,:,:]
        s1 = s1[:,None,:,:,:]
        b = torch.cat([b, s1], dim=1)
        b = self.block9(b)

        return b
