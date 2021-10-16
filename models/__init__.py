from .lowresdensenet import lowres_densenet121, lowres_densenet161, lowres_densenet169
from .lowresgooglenet import lowres_googlenet
from .lowresinception import lowres_inception_v3
from .lowresmobilenetv2 import lowres_mobilenet_v2
from .lowresresnet import lowres_resnet14, lowres_resnet18, lowres_resnet18_noresidual, lowres_resnet34, \
    lowres_resnet50, lowres_resnet101, lowres_resnet152
from .lowresvgg import lowres_vgg11_bn, lowres_vgg13_bn, lowres_vgg16_bn, lowres_vgg19_bn, lowres_vgg11, lowres_vgg13, \
    lowres_vgg16, lowres_vgg19
from .lowresresnet9 import lowres_resnet9
from .lowresalexnet import lowres_alexnet

all_classifiers = {
    "lowres_vgg11_bn": lowres_vgg11_bn,
    "lowres_vgg13_bn": lowres_vgg13_bn,
    "lowres_vgg16_bn": lowres_vgg16_bn,
    "lowres_vgg19_bn": lowres_vgg19_bn,
    "lowres_vgg11": lowres_vgg11,
    "lowres_vgg13": lowres_vgg13,
    "lowres_vgg16": lowres_vgg16,
    "lowres_vgg19": lowres_vgg19,
    "lowres_resnet14": lowres_resnet14,
    "lowres_resnet18": lowres_resnet18,
    "lowres_resnet18_noresidual": lowres_resnet18_noresidual,
    "lowres_resnet34": lowres_resnet34,
    "lowres_resnet50": lowres_resnet50,
    "lowres_resnet101": lowres_resnet101,
    "lowres_resnet152": lowres_resnet152,
    "lowres_resnet9": lowres_resnet9,
    "lowres_densenet121": lowres_densenet121,
    "lowres_densenet161": lowres_densenet161,
    "lowres_densenet169": lowres_densenet169,
    "lowres_mobilenet_v2": lowres_mobilenet_v2,
    "lowres_googlenet": lowres_googlenet,
    "lowres_inception_v3": lowres_inception_v3,
    "lowres_alexnet": lowres_alexnet
}


def get_model(name):
    return all_classifiers.get(name)
