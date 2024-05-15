
import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import cv2
# from ..model_training.cross_entropy_pre_training.cross_entropy_model import FBankCrossEntropyNet
from MiniFASNet import MiniFASNetV2

self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
kernel_size = ((80 + 15) // 16, (80 + 15) // 16)
# MODEL_PATH = r'2.7_80x80_MiniFASNetV2.pth'
model_instance = MiniFASNetV2(conv6_kernel=kernel_size).to(self.device)

state_dict = torch.load(MODEL_PATH, map_location=self.device)
# state_dict = torch.load(model_path, map_location=self.device)
keys = iter(state_dict)
first_layer_name = keys.__next__()
if first_layer_name.find('') >= 0:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name_key = key[7:]
        new_state_dict[name_key] = value
    model_instance.load_state_dict(new_state_dict)
else:
    model_instance.load_state_dict(state_dict)


# model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
# model_instance = model_instance.double()
model_instance.eval()
# model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

img = cv2.imread(r"Test image.JPG")
# img = cv2.resize(img, (80,80))
test_transform = trans.Compose([
    trans.ToTensor(),
])
img = test_transform(img)
dummy_input = img.unsqueeze(0).to(device)
# dummy_input = torch.from_numpy(img.reshape(1, -1)).float().to(device)
# dummy_output = model_pytorch(dummy_input)
# print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_instance, dummy_input, 'model_simple.onnx', input_names=['test_input'], output_names=['test_output'])
