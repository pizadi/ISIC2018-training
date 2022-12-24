import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from BaseModel import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding="same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, (3, 3), padding="same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            SEBlock(out_c)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, (1, 1)),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.fc(y)
        return x*y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1)),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU())
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding="same", dilation=6, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding="same", dilation=12, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding="same", dilation=18, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

        self.block = nn.Sequential(
            nn.Conv2d(5*out_c, out_c, (1, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU())
        

    def forward(self, X):
        X = X.to(self.device)
        X1 = self.block_1(torch.mean(X, (2, 3), keepdim=True))
        X1 = X1.repeat((1, 1, X.shape[2], X.shape[3]))
        X2 = self.block_2(X)
        X3 = self.block_3(X)
        X4 = self.block_4(X)
        X5 = self.block_5(X)
        
        y = self.block(torch.cat([X1, X2, X3, X4, X5], dim=1))
        return y

class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()

        network = vgg19(pretrained=True)

        self.b1 = network.features[:4]
        self.b2 = network.features[4:9]
        self.b3 = network.features[9:18]
        self.b4 = network.features[18:27]
        self.b5 = network.features[27:36]

    def forward(self, X):
        x1 = self.b1(X) 
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        return x5, [x4, x3, x2, x1]

class decoder1(nn.Module):
    def __init__(self):
        super(decoder1, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = ConvBlock(64+512, 256)
        self.c2 = ConvBlock(512, 128)
        self.c3 = ConvBlock(256, 64)
        self.c4 = ConvBlock(128, 32)

    def forward(self, X, skip):
        s1, s2, s3, s4 = skip

        X = self.up(X)
        X = torch.cat([X, s1], axis=1)
        X = self.c1(X)

        X = self.up(X)
        X = torch.cat([X, s2], axis=1)
        X = self.c2(X)

        X = self.up(X)
        X = torch.cat([X, s3], axis=1)
        X = self.c3(X)

        X = self.up(X)
        X = torch.cat([X, s4], axis=1)
        X = self.c4(X)

        return X

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.pool4 = nn.MaxPool2d((2, 2))

        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.c4 = ConvBlock(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool1(x1)

        x2 = self.c2(p1)
        p2 = self.pool2(x2)

        x3 = self.c3(p2)
        p3 = self.pool3(x3)

        x4 = self.c4(p3)
        p4 = self.pool4(x4)

        return p4, [x4, x3, x2, x1]

class decoder2(nn.Module):
    def __init__(self):
        super(decoder2, self).__init__()

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = ConvBlock(832, 256)
        self.c2 = ConvBlock(640, 128)
        self.c3 = ConvBlock(320, 64)
        self.c4 = ConvBlock(160, 32)

    def forward(self, x, skip1, skip2):

        x = self.up1(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up2(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up4(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x

class DoubleUNet(nn.Module):
  def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
    super(DoubleUNet, self).__init__()

    if (device is None):
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    if (learning_rate is None):
      self.learning_rate = 1e-4
    else:
      self.learning_rate = learning_rate

    
    self.e1 = encoder1()
    self.a1 = ASPP(512, 64)
    self.d1 = decoder1()
    self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
    self.sigmoid = nn.Sigmoid()

    self.e2 = encoder2()
    self.a2 = ASPP(256, 64)
    self.d2 = decoder2()
    self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    for param in self.e1.parameters():
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
    x0 = X
    X, skip1 = self.e1(X)
    X = self.a1(X)
    X = self.d1(X, skip1)
    y1 = torch.sigmoid(self.y1(X))

    input_x = x0 * self.sigmoid(y1)
    X, skip2 = self.e2(input_x)
    X = self.a2(X)
    X = self.d2(X, skip1, skip2)
    y2 = torch.sigmoid(self.y2(X))

    return y1, y2

  def fit(self, X, y):
    X, y = X.to(self.device), y.to(self.device)
    torch.cuda.empty_cache()
    pred1, pred2 = self(X)
    self.optimizer.zero_grad()
    loss = self.loss_fn(pred1, y) + self.loss_fn(pred2, y)
    loss.backward()
    self.optimizer.step()
    pred1, X = None, None
    torch.cuda.empty_cache()
    y = (y > 0.5)
    pred2 = (pred2 > 0.5)
    numt = torch.numel(y[0])
    TP = torch.sum(torch.bitwise_and(y, pred2)).item() / numt
    TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred2))).item() / numt
    FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred2))).item() / numt
    FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred2)).item() / numt
    return (loss.item(), torch.tensor([TP, FP, FN, TN]))

  def test(self, X, y):
      X, y = X.to(self.device), y.to(self.device)
      pred1, pred2 = self(X)
      loss = self.loss_fn(pred1, y) + self.loss_fn(pred2, y)
      y = (y > 0.5)
      pred2 = (pred2 > 0.5)
      numt = torch.numel(y[0])
      TP = torch.sum(torch.bitwise_and(y, pred2)).item() / numt
      TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred2))).item() / numt
      FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred2))).item() / numt
      FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred2)).item() / numt
      return (loss.item(), torch.tensor([TP, FP, FN, TN]))