"""
This module contains the VGG based convolution model.
"""
import torch.nn as nn

CONV_CONFIG = {
    'vgg6': [64, 64, 'M', 128, 128, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 'M', 512, 512, 512, 'M']
}

FC_CONFIG = {
    'vgg6': [100],
    'vgg16': [2048, 2048]
}

ACTIVATION_CONFIG = {
    'vgg6': nn.ReLU,
    'vgg16': nn.LeakyReLU
}

class CustomVGG(nn.Module):
    """
        Custom implemnattion of VGG16 model architecture with varying feature (conv) layers
        and classification (fully connected) layers. Also uses different activation functions
        like leakyReLU when specified in either feature or classification layers.
    """
    def __init__(self, features, classifier_layers, n_in=28, n_out=10, num_channels=1, init_weights=True, activation_fn=nn.ReLU):
        super(CustomVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            *classifier_layers,
            nn.Linear(classifier_layers[-3].out_features, n_out))
        if init_weights:
            self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn):
        non_linearity = 'leaky_relu' if activation_fn == nn.LeakyReLU else 'relu'
        print(f"Weight initialization for nonlinearity :{non_linearity}")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=non_linearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=non_linearity)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))
        x = self.classifier(x)
        return x


def make_conv_layers(config, activation, drop_prob=0.5, batch_norm=False, stitch_together=True):
    assert activation in [nn.ReLU, nn.LeakyReLU]

    # conv layers have a higher keep prob (lower drop rate) than fc layers [as per paper]
    keep_prob = min((1-drop_prob)+0.3, 1.0)
    drop_prob = (1 - keep_prob)

    activation_layer = activation(inplace=True,
                                  **(dict(negative_slope=0.2)
                                     if activation == nn.LeakyReLU else {}))
    layers = []
    # always assume RGB channels, so size up a black white image from 1 channel
    # using torchvision.transforms
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation_layer, nn.Dropout(p=drop_prob)]
            else:
                layers += [conv2d, activation_layer, nn.Dropout(p=drop_prob)]
            in_channels = v
    return nn.Sequential(*layers) if stitch_together else layers


def make_fc_layers(config, activation, in_features, drop_prob=0.5, stitch_together=False):
    assert activation in [nn.ReLU, nn.LeakyReLU]

    activation_layer = activation(inplace=True,
                                  **(dict(negative_slope=0.2)
                                     if activation == nn.LeakyReLU else {}))

    layers = []
    prev_layer_size = in_features
    for layer_size in config:
        layers += [nn.Linear(prev_layer_size, layer_size),
                   activation_layer,
                   nn.Dropout(p=drop_prob)]
        prev_layer_size = layer_size

    return nn.Sequential(*layers) if stitch_together else layers

def vgg_model(n_in, n_out, drop_rate, model_type: str, num_channels, **kwargs):
    conv_config = CONV_CONFIG[model_type]
    features = make_conv_layers(conv_config,
                                ACTIVATION_CONFIG[model_type],
                                drop_prob=drop_rate)
    # last layer is Maxpool, look at layer before - indicates the number of conv filters
    # for the avgpool applied with target (H,W) = (7,7)
    input_features_to_fc = conv_config[-2] * 7 * 7
    classifier_layers = make_fc_layers(FC_CONFIG[model_type],
                                       ACTIVATION_CONFIG[model_type],
                                       input_features_to_fc,
                                       drop_prob=drop_rate)
    return CustomVGG(features, classifier_layers, n_in=n_in,
                     n_out=n_out, num_channels=num_channels,
                     activation_fn=ACTIVATION_CONFIG[model_type])
