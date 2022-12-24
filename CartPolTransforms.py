import torch
import numpy as np
import cv2 as cv

def centerofmass(img):
    if img.shape[0] > 1:
        raise ValueError("Only single-channel images allowed")
    i, j = img.shape[1:3]
    if img.dtype is torch.int:
        img = torch.clamp(img, min=0, max=255).float()/255
    elif torch.max(img) > 1.:
        img = torch.clamp(img, min=0)/torch.max(img)

    imap = (torch.arange(i)[None,:,None]).repeat(1, 1, j)
    jmap = (torch.arange(j)[None,None,:]).repeat(1, i, 1)
    sum = torch.sum(img)

    return torch.sum(imap*img)/sum, torch.sum(jmap*img)/sum

def cart2pol(img, center):
  if img.shape[0] == 1:
    input_img = img[0,:,:].numpy()
  else:
    input_img = img.transpose(0,1).transpose(1,2).numpy()
  value = np.sqrt(((input_img.shape[0]/2.0)**2.0)+((input_img.shape[1]/2.0)**2.0))
  polar_image = cv.linearPolar(input_img, center, value, cv.WARP_FILL_OUTLIERS)
  polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
  if img.shape[0] == 1:
    return torch.tensor(polar_image)[None,:,:]
  else:
    return torch.tensor(polar_image).transpose(0,1).transpose(1,2)