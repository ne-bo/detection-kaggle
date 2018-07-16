import sys
from os import path
import torch.nn.functional as F

# https://github.com/marvis/pytorch-yolo2
from utils.util import get_ids_dict

sys.path.append(path.abspath('/home/natasha/PycharmProjects/pytorch-yolo2/'))
from region_loss import RegionLoss

def my_loss(y_input, y_target):
    return RegionLoss(y_input, y_target)
