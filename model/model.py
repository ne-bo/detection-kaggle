import sys

import torch

from base import BaseModel
from os import path


# https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
sys.path.append(path.abspath('/home/natasha/PycharmProjects/pytorch-yolo-v3/'))
from darknet import Darknet


class NatashaDetection(BaseModel):
    def __init__(self, config):
        super(NatashaDetection, self).__init__(config)
        self.config = config
        print("Loading network.....")
        self.net = Darknet('/home/natasha/PycharmProjects/pytorch-yolo-v3/cfg/yolov3.cfg').cuda()
        self.net.load_weights('/home/natasha/detection_kaggle/yolov3-320.weights')
        print("Network successfully loaded")

    def forward(self, x):
        x = self.net(x, CUDA=torch.cuda.is_available())
        return x
