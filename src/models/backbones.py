"""Backbone model loaders (VGG19, DenseNet201, MobileNetV2)"""
import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(name, num_classes=6, pretrained=True):
    name = name.lower()
    if name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        # replace classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        feat_dim = 512
    elif name == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        feat_dim = 1920
    elif name == 'mobilenet_v2' or name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        feat_dim = 1280
    else:
        raise ValueError(f'Unknown backbone {name}')
    return model, feat_dim
