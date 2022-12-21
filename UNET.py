import torch
from torch import nn as nn
import torchvision
from BaseModel import BaseModel

class UNET(BaseModel):
  def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
    super(UNET, self).__init__()
    
    if (device is None):
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    if (learning_rate is None):
      self.learning_rate = 1e-4
    else:
      self.learning_rate = learning_rate

    self.input = nn.Conv2d(3, 4, (1, 1),padding="same")
    
    self.enc1 = self.encblock(4, 8)
    self.enc2 = self.encblock(8, 16)
    self.enc3 = self.encblock(16, 32)
    self.enc4 = self.encblock(32, 64)
    
    self.dec4 = self.decblock(64, 32)
    self.dec3 = self.decblock(64, 16)
    self.dec2 = self.decblock(32, 8)
    self.dec1 = self.decblock(16, 4)
    
    self.output = nn.Conv2d(8, 1, (1, 1), padding="same")

    if (loss_fn is None):
      self.loss_fn = nn.BCELoss()
    else:
      self.loss_fn = loss_fn
    
    if (optimizer is None):
      self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    else:
      self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)

    
    self = self.to(self.device)

  def encblock(self, in_features, out_features):
    block = nn.Sequential(
        nn.Conv2d(in_features, out_features, (3, 3), padding="same"),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
        nn.Conv2d(out_features, out_features, (1, 1), padding="same"),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
        nn.MaxPool2d((2, 2))
        )
    block = block.to(self.device)
    return block

  def decblock(self, in_features, out_features):
      block = nn.Sequential(
          nn.ConvTranspose2d(in_features, out_features, 2, stride=(2, 2)),
          nn.BatchNorm2d(out_features),
          nn.ReLU(),
          nn.Conv2d(out_features, out_features, (5, 5), padding="same"),
          nn.BatchNorm2d(out_features),
          nn.ReLU()
          )
      block = block.to(self.device)
      return block

  def forward(self, X):
      X = self.input(X)
      h1 = self.enc1(X)
      h2 = self.enc2(h1)
      h3 = self.enc3(h2)
      h4 = self.enc4(h3)
      h4 = self.dec4(h4)
      h3 = torch.cat((h3, h4), dim=1)
      h4 = None
      h3 = self.dec3(h3)
      h2 = torch.cat((h2, h3), dim=1)
      h3 = None
      h2 = self.dec2(h2)
      h1 = torch.cat((h1, h2), dim=1)
      h2 = None
      h1 = self.dec1(h1)
      X = torch.cat((X, h1), dim=1)
      h1 = None
      X = self.output(X)
      X = torch.sigmoid(X)
      return X
