import torch
import cv2
import numpy as np


def read_img(path, transforms, device):
    img = transforms(cv2.imread(path)).to(device)
    img = torch.unsqueeze(img, dim=0)
    return img


def save_img(img, path_to_save):
    img = torch.squeeze(img, dim=0)
    img = img.permute(1,2, 0).detach().to('cpu').numpy()
    img = img * 255
    img  = img.astype(np.int32)
    cv2.imwrite(path_to_save, img)
