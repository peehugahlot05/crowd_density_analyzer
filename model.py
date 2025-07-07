import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512,256,128,64]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        vgg16 = models.vgg16(pretrained=True)
        self._initialize_weights()
        self._load_pretrained_weights(vgg16)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1 if not dilation else 2, dilation=1 if not dilation else 2)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, vgg):
        frontend_dict = self.frontend.state_dict()
        vgg_dict = vgg.features.state_dict()
        pretrained_dict = {k: v for k, v in vgg_dict.items() if k in frontend_dict}
        frontend_dict.update(pretrained_dict)
        self.frontend.load_state_dict(frontend_dict)

    def _initialize_weights(self):
        for m in self.backend.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
