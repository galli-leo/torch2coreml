import sys
import os
sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../torch2coreml"
)

from _torch_converter import convert
import _layers as layers
from torch.utils.serialization import load_lua
import torch as th
import torch.tensor
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable

def convert_unknown(builder, name, layer, input_names, output_names):
    print("!! No converter yet for layer: "+name)
    return output_names

coreml_model = convert(
        "openface.t7",
        [(3,96,96)],
        image_input_names=['input'],
        output_shapes=[[128]],
        mode=None,
        unknown_layer_converter_fn=convert_unknown
    )

coreml_model.author = 'Leonardo Galli feat. OpenFace'
coreml_model.license = 'Free for personal or research use'
coreml_model.save("openface.mlmodel")
