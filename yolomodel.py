## yolov3 model from scratch
import torch 
import torch.nn as nn
import os
import cv2
import torch.nn.functional as F

def conf(config_path):
    with open(config_path, 'r') as file:
        content = file.read().splitlines()
    layers = []
    current_layer = {} 
    for line in content:
        line = line.strip()        
        if not line or line.startswith('#'):
            continue     
        if '=' in line:
            key, value = line.split('=')
            current_layer[key.strip()] = value.strip()
        else:
            if current_layer:
                layers.append(current_layer)
            current_layer = {'type': line.strip('[]')}   
    if current_layer:  # Append the last layer
        layers.append(current_layer)
    
    return layers

class YOLOLayer(nn.Module):
    def __init__(self, anchors,num_classes):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_clasees = num_classes
        self.num_anchors = len(anchors)
        self.obj_confidence = 1

    def forward(self, x):
        batch_size, _, grid_size, _ = x.size()
        grid_size = x.size(2)
        prediction = x.view(batch_size, self.num_anchors, self.num_classes+5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2)

        for i in range(5):
            prediction[..., i] = torch.sigmoid(prediction[..., i])
        prediction[..., 5] = prediction[..., 5:] 
        return prediction
def create_modules(layers, input_channels):
    modules = []
    output_channels = input_channels
    filterTracker = [input_channels]
    for layer in layers:
        if layer["type"] == "convolution":
            filter  = int(layer["filter"])
            stride  = int(layer.get("stride",1))
            pad = int(layer["pad"])
            kernel_size = int(layer["size"])
            if pad:
                padding = (kernel_size -1 )//2
            else:
                padding = 0
            
            activation =layer.get("activation","linear")
            try:
                bn = int(layer["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True
            conv = nn.Conv2d(output_channels, filter, kernel_size, stride, padding, bias=bias)
            modules.append(conv)

            if bn:
                modules.append(nn.BatchNorm2d(filter))
                
            if activation == "leaky":
                modules.append(nn.LeakyReLU(0.1, inplace=True))
            output_channels = filter

        elif layer["type"] == "upsample":
            modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            output_channels = output_channels

        elif layer["type"] == "route":
            layers = layer["layers"].split(',')
            start = int(layer["layers"][0])
            end = int(layer["layers"][1]) if len(layer["layers"]) > 1 else 0

            if start > 0:
                start = start - len(filterTracker)
            if end > 0:
                end = end - len(filterTracker)
            if end == 0:
                output_channels = filterTracker[start]

            else:
                output_channels = filterTracker[start] + filterTracker[end]
            
        elif layer["type"] == "shortcut":
            modules.append(nn.Identity())
            output_channels = output_channels
        elif layer["type"] == "yolo":
            mask = layer["mask"].split(',')
            mask = [int(x) for x in mask]
            anchors = layer["anchors"].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            modules.append(YOLOLayer(anchors))
            output_channels = output_channels
        filterTracker.append(output_channels)
    return nn.Sequential(modules)




