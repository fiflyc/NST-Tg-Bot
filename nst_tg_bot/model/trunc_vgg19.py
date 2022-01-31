import torch
import torch.nn as nn
import torchvision.models as models


class TruncVGG19(nn.Module):
    def __init__(self, last_layer, pretrained=True):
        super().__init__()

        self.__vgg19 = models.vgg19(pretrained=pretrained)
        for p in self.__vgg19.parameters():
            p.requires_grad = False
        self.__output = 0
        self.__vgg19.features[last_layer].register_forward_hook(self.__hook)
    
    def __hook(self, module, input, output):
        self.__output = output

    def forward(self, x):
        self.__vgg19(x)

        return self.__output