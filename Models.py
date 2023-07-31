
# TODO Model/Architecture Construction

'''
Implementation (in accordance to given 'paper' parametric specifications) of:
    DenseNet-BC architecture
        (with L=100, k=12 later on)
    VGG Architecture
        (with configuration 'E', 19 weight layers later on)
'''

import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------------------------------------------------------

# TODO IMPLEMENT PYTHON CLASS FOR DENSENET-BC ARCHITECTURE
'''
    "DenseNet-BC model architecture with L = 100; k = 12"
'''
class DenseNetBC(nn.Module):
    # Including proposed L=100 and k=12, as well as number of classes (10 for CIFAR-10 and 43 for GTSRB)
    # k=12 -> 'Growth Rate' to be included in our DenseNet-BC model per 'paper'
    def __init__(self, num_classes, L=100, k=12):
        super(DenseNetBC, self).__init__()
        self.densenet = models.densenet121(
                                            weights=None,
                                            num_classes=num_classes,
                                            )
    name = 'DenseNet-BC'  # to display during training and testing performance announcements in the console

    def forward(self, x):
        return self.densenet(x)

    '''
    Any required 'Instance' of a 'Model' of this 'DenseNet-BC Architecture' will be achieved such as following:
    1) Define number of classes of the specific dataset of intake
        num_classes = 10 (e.g.), perhaps num_classes_cifar or num_classes_gtsrb would already be present to work with
    2) Simply call the constructor method:
        model = DenseNetBC(num_classes, L=100, k=12) -> This 'model' is a new, fresh 'instance' that we can work with
    '''

# ---------------------------------------------------------------------------------------------------------------------

# TODO IMPLEMENT PYTHON CLASS FOR VGG ARCHITECTURE
'''
    "Configuration ’E’ VGG architecture with 19 weight layers"
'''
class VGG(nn.Module):
    # No additional adjustments needed, as vgg19 already corresponds to configuration 'E' with 19 weight layers
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.vgg = models.vgg19(weights=None, num_classes=num_classes)
    name = 'VGG19'  # to display during training and testing performance announcements in the console

    def forward(self, x):
        return self.vgg(x)

    '''
    Any required 'Instance' of a 'Model' of this 'VGG Architecture' will be achieved such as following:
    1) Define number of classes of the specific dataset of intake
        num_classes = 10 (e.g.), perhaps num_classes_cifar or num_classes_gtsrb would already be present to work with
    2) Simply call the constructor method:
        model = VGG(num_classes) -> This 'model' is now a new, fresh 'instance' that we can work with
    '''