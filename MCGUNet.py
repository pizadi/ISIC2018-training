import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from BaseModel import BaseModel

class UNetRes50(BaseModel):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
        super(UNetRes50, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate
            
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), padding="same")
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding="same")
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), padding="same")
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout3 = nn.Dropout(0.5)

        #D1
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), padding="same")
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), padding="same")\
        self.dropout4 = nn.Dropout(0.5)

        #D2
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), padding="same")
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), padding="same")\
        self.dropout5 = nn.Dropout(0.5)

        #D3
        self.conv6_1 = nn.Conv2d(1024, 512, (3, 3), padding="same")
        self.conv6_2 = nn.Conv2d(512, 512, (3, 3), padding="same")\
        self.dropout6 = nn.Dropout(0.5)

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


    def forward(self, X):
        X = X.to(self.device)

        s1 = self.e1(X)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        
        s4 = self.u1(s4)
        s3 = self.c1(torch.cat([s3, s4], dim=1))
        s4 = None

        s3 = self.u2(s3)
        s2 = self.c2(torch.cat([s2, s3], dim=1))
        s3 = None

        s2 = self.u3(s2)
        s1 = self.c3(torch.cat([s1, s2], dim=1))
        s2 = None

        s1 = self.u4(s1)
        X = self.c4(torch.cat([X, s1], dim=1))
        s1 = None

        X = torch.sigmoid(self.out(X))

        return X
