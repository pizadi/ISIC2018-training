import torch
from torch import nn as nn
from torch import sqrt, pow
import torchvision

class LossNormModel(nn.Module):
  
    def fit(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        torch.cuda.empty_cache()
        pred = self(X)
        self.optimizer.zero_grad()
        t = torch.sum(y)/torch.numel(y)
        loss1 = self.loss_fn(pred*y, y)/t
        loss2 = self.loss_fn(pred*(1-y), y)/(1-t)
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        y = (y > 0.5)
        pred = (pred > 0.5)
        numt = torch.numel(y[0])
        TP = torch.sum(torch.bitwise_and(y, pred)).item() / numt
        TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred))).item() / numt
        FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred))).item() / numt
        FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred)).item() / numt
        return (loss, torch.tensor([TP, FP, FN, TN]))

    def test(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        pred = self(X)
        t = torch.sum(y)/torch.numel(y)
        loss1 = self.loss_fn(pred*y, y)/t
        loss2 = self.loss_fn(pred*(1-y), y)/(1-t)
        loss = loss1 + loss2
        y = (y > 0.5)
        pred = (pred > 0.5)
        numt = torch.numel(y[0])
        TP = torch.sum(torch.bitwise_and(y, pred)).item() / numt
        TN = torch.sum(torch.bitwise_not(torch.bitwise_or(y, pred))).item() / numt
        FN = torch.sum(torch.bitwise_and(y, torch.bitwise_not(pred))).item() / numt
        FP = torch.sum(torch.bitwise_and(torch.bitwise_not(y), pred)).item() / numt
        return (loss, torch.tensor([TP, FP, FN, TN]))