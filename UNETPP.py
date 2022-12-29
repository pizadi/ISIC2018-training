import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
from BaseModel import BaseModel

class UNetPP(nn.Module):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
        super(UNetPP, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate

        self.scale = 32

        self.input = nn.Conv2d(3, self.scale, (1, 1),padding="same")

        self.b0_0 = convblock(self.scale)
        self.b0_1 = convblock(self.scale)
        self.up0_1 = nn.ConvTranspose2d(self.scale, self.scale, 2, stride=(2, 2))
        self.b0_2 = convblock(self.scale)
        self.up0_2 = nn.ConvTranspose2d(self.scale, self.scale, 2, stride=(2, 2))
        self.b0_3 = convblock(self.scale)
        self.up0_2 = nn.ConvTranspose2d(self.scale, self.scale, 2, stride=(2, 2))
        self.b0_4 = convblock(self.scale)
        self.up0_2 = nn.ConvTranspose2d(self.scale, self.scale, 2, stride=(2, 2))

        self.b1_0 = convblock(self.scale * 2)
        self.b1_1 = convblock(self.scale * 2)
        self.up1_1 = nn.ConvTranspose2d(self.scale * 2, self.scale * 2, 2, stride=(2, 2))
        self.b1_2 = convblock(self.scale * 2)
        self.up1_2 = nn.ConvTranspose2d(self.scale * 2, self.scale * 2, 2, stride=(2, 2))
        self.b1_3 = convblock(self.scale * 2)
        self.up1_3 = nn.ConvTranspose2d(self.scale * 2, self.scale * 2, 2, stride=(2, 2))

        self.b2_0 = convblock(self.scale * 5)
        self.b2_1 = convblock(self.scale * 5)
        self.up2_1 = nn.ConvTranspose2d(self.scale * 5, self.scale * 5, 2, stride=(2, 2))
        self.b2_2 = convblock(self.scale * 5)
        self.up2_2 = nn.ConvTranspose2d(self.scale * 5, self.scale * 5, 2, stride=(2, 2))

        self.b3_0 = convblock(self.scale * 13)
        self.b3_1 = convblock(self.scale * 13)
        self.up3_1 = nn.ConvTranspose2d(self.scale * 13, self.scale * 13, 2, stride=(2, 2))

        self.b4_0 = convblock(self.scale * 34)

        self.out1 = nn.Conv2d(scale*2, 1, (1, 1))
        self.out2 = nn.Conv2d(scale*5, 1, (1, 1))
        self.out3 = nn.Conv2d(scale*13, 1, (1, 1))
        self.out4 = nn.Conv2d(scale*34, 1, (1, 1))

        if (loss_fn is None):
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = loss_fn

        if (optimizer is None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)


        self = self.to(self.device)

    def convblock(self, features):
        block = nn.Sequential(
            nn.Conv2d(features, features, (3, 3), padding="same"),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, (3, 3), padding="same"),
            nn.BatchNorm2d(features),
            nn.ReLU()
            )
        block = block.to(self.device)
        return block

  def decblock(self, in_features, out_features):
      block = nn.Sequential(
          nn.ConvTranspose2d(in_features, out_features, 2, stride=(2, 2)),,
            nn.MaxPool2d((2, 2))

    def forward(self, X):
        X = X.to(self.device)
        X = self.input(X)

        x0_0 = self.b0_0(X)
        x0_1 = self.b0_1(F.max_pool2d(x0, (2, 2), (2, 2)))
        x0_2 = self.b0_2(F.max_pool2d(x1, (2, 2), (2, 2)))
        x0_3 = self.b0_3(F.max_pool2d(x2 (2, 2), (2, 2)))
        x0_4 = self.b0_4(F.max_pool2d(x3, (2, 2), (2, 2)))

        x1_0 = self.b1_0(torch.cat([x0_0, self.up0_1(x0_1)], dim=1))
        x1_1 = self.b1_1(torch.cat([x0_1, self.up0_2(x0_2)], dim=1))
        x1_2 = self.b1_2(torch.cat([x0_2, self.up0_3(x0_3)], dim=1))
        x1_3 = self.b1_3(torch.cat([x0_3, self.up0_4(x0_4)], dim=1))
        
        x2_0 = self.b2_0(torch.cat([x0_0, x1_0, self.up1_1(x1_1)], dim=1))
        x2_1 = self.b2_1(torch.cat([x0_1, x1_1, self.up1_2(x1_2)], dim=1))
        x2_2 = self.b2_2(torch.cat([x0_2, x1_2, self.up1_3(x1_3)], dim=1))

        x3_0 = self.b3_0(torch.cat([x0_0, x1_0, x2_0, self.up2_1(x2_1)], dim=1))
        x3_1 = self.b3_1(torch.cat([x0_1, x1_1, x2_1, self.up2_2(x2_2)], dim=1))

        x4_0 = self.b4_0(torch.cat([x0_0, x1_0, x2_0,  x3_0, self.up3_1(x3_1)], dim=1))

        y1 = torch.sigmoid(self.out1(x0_1))
        y2 = torch.sigmoid(self.out2(x0_2))
        y3 = torch.sigmoid(self.out3(x0_3))
        y4 = torch.sigmoid(self.out4(x0_4))

        return y1, y2, y3, y4

    def fit(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        torch.cuda.empty_cache()
        pred1, pred2, pred3, pred4 = self(X)
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred1, y) + self.loss_fn(pred2, y) + self.loss_fn(pred3, y) + self.loss_fn(pred4, y)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        y = (y > 0.5)
        pred = (pred4 > 0.5)
        numt = torch.numel(y[0])
        TP = torch.sum(torch.bitwise_and(y, pred)).item() / numt
        TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred))).item() / numt
        FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred))).item() / numt
        FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred)).item() / numt
        return (loss, torch.tensor([TP, FP, FN, TN]))

    def test(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        pred1, pred2, pred3, pred4 = self(X)
        loss = self.loss_fn(pred1, y) + self.loss_fn(pred2, y) + self.loss_fn(pred3, y) + self.loss_fn(pred4, y)
        y = (y > 0.5)
        pred = (pred > 0.5)
        numt = torch.numel(y[0])
        TP = torch.sum(torch.bitwise_and(y, pred)).item() / numt
        TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred))).item() / numt
        FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred))).item() / numt
        FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred)).item() / numt
        return (loss, torch.tensor([TP, FP, FN, TN]))