
# TODO Dataset Preparation

'''
Preparation of CIFAR-10 and GTSRB datasets to be utilized in the experiment phase
Downloading and/or extracting training and testing sets for each of our datasets
'clean' and 'poisoned' variants for both datasets will be set up and preprocessed
'''

import torch
import torch.utils.data
from torchvision import datasets, transforms
# Importing our own file for the 'poisoning process'
import BackdoorTrigger

# ---------------------------------------------------------------------------------------------------------------------

def prepare(apply_trigger):

    # TODO DEFINE THE TRANSFORMATION FOR CIFAR-10
    # Defining the required transformation for our CIFAR-10 datasets TODO TRANSFORM FOR CIFAR-10
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # mean = 0.5, std. dev. = 0.5
    ])

    # TODO DOWNLOAD & INSTANTIATE 'CLEAN' & 'TO BE POISONED' CIFAR-10 TRAINING DATASETS
    print("CIFAR-10 Training Dateset (clean):")
    # Loading and instantiating 'clean' CIFAR-10 training dataset (via torchvision.datasets) TODO CLEAN CIFAR-10 TRAIN
    cifar_clean_train = datasets.CIFAR10(root='./datasets/cifar/clean/train',
                                         train=True, download=True, transform=transform_cifar)
    print("CIFAR-10 Training Dateset (poison):")
    # Loading and instantiating (to be) 'poisoned' CIFAR-10 training dataset TODO (TO BE) POISONED CIFAR-10 TRAIN
    cifar_poisoned_train = datasets.CIFAR10(root='./datasets/cifar/poisoned/train',
                                            train=True, download=True, transform=transform_cifar)

    # TODO DOWNLOAD & INSTANTIATE 'CLEAN' & 'TO BE POISONED' CIFAR-10 TESTING DATASETS
    print("CIFAR-10 Testing Dateset (clean):")
    #Loading and instantiating 'clean' CIFAR-10 testing dataset (via torchvision.datasets) TODO CLEAN CIFAR-10 TEST
    cifar_clean_test = datasets.CIFAR10(root='./datasets/cifar/clean/test',
                                        train=False, download=True, transform=transform_cifar)
    print("CIFAR-10 Testing Dateset (poison):")
    #Loading and instantiating (to be) 'poisoned' CIFAR-10 testing dataset TODO (TO BE) POISONED CIFAR-10 TEST
    cifar_poisoned_test = datasets.CIFAR10(root='./datasets/cifar/poisoned/test',
                                           train=False, download=True, transform=transform_cifar)

    # -----------------------------------------------------------------------------------------------------------------

    # TODO DEFINE THE TRANSFORMATION FOR GTSRB
    # Defining the required transformation for our GTSRB datasets TODO TRANSFORM FOR GTSRB
    transform_gtsrb = transforms.Compose([
        transforms.Resize((32, 32)),  # to make GTSRB instances dimensionally compatible with our models TODO RESIZE
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # mean = 0.5, std. dev. = 0.5 for each image
    ])

    # TODO DOWNLOAD & INSTANTIATE 'CLEAN' & 'TO BE POISONED' GTSRB TRAINING DATASETS
    print("GTSRB Training Dateset (clean):")
    # Loading and instantiating 'clean' GTSRB training dataset TODO CLEAN GTSRB TRAIN
    gtsrb_clean_train = datasets.GTSRB(root='./datasets/gtsrb/clean/train',
                                       split='train', download=True, transform=transform_gtsrb)
    print("Downloaded and verified")
    print("GTSRB Training Dateset (poison):")
    # Loading and instantiating (to be 'poisoned') GTSRB training dataset TODO (TO BE) POISONED GTSRB TRAIN
    gtsrb_poisoned_train = datasets.GTSRB(root='./datasets/gtsrb/poisoned/train',
                                          split='train', download=True, transform=transform_gtsrb)
    print("Downloaded and verified")

    # TODO DOWNLOAD & INSTANTIATE 'CLEAN' & 'TO BE POISONED' GTSRB TESTING DATASETS
    print("GTSRB Testing Dateset (clean):")
    # Loading and instantiating 'clean' GTSRB testing dataset TODO CLEAN GTSRB TEST
    gtsrb_clean_test = datasets.GTSRB(root='./datasets/gtsrb/clean/test',
                                      split='test', download=True, transform=transform_gtsrb)
    print("Downloaded and verified")
    print("GTSRB Testing Dateset (poison):")
    # Loading and instantiating (to be 'poisoned') GTSRB testing dataset TODO (TO BE) POISONED GTSRB TEST
    gtsrb_poisoned_test = datasets.GTSRB(root='./datasets/gtsrb/poisoned/test',
                                         split='test', download=True, transform=transform_gtsrb)
    print("Downloaded and verified")

    # -----------------------------------------------------------------------------------------------------------------

    print()
    # BREAKPOINT 1: Verification of Datasets
    # breakpoint()

    # TODO (1.2) Checking poisoned datasets (CIFAR-10 AND GTSRB) (TRAIN AND TEST), apply backdoor pattern if required

    if apply_trigger:
        BackdoorTrigger.apply_cifar("train", cifar_poisoned_train, 2)  # See 'BackdoorTrigger.py' for details
    else:
        print("Skipping backdoor pattern application for CIFAR-10 TRAINING set (assuming dataset previously poisoned)")

    if apply_trigger:
        BackdoorTrigger.apply_cifar("test", cifar_poisoned_test, None)  # See 'BackdoorTrigger.py' for details
    else:
        print("Skipping backdoor pattern application for CIFAR-10 TESTING set (assuming dataset previously poisoned)")

    if apply_trigger:
        BackdoorTrigger.apply_gtsrb("train", gtsrb_poisoned_train, 2)  # See 'BackdoorTrigger.py' for details
    else:
        print("Skipping backdoor pattern application for GTSRB TRAINING set (assuming dataset previously poisoned)")

    if apply_trigger:
        BackdoorTrigger.apply_gtsrb("test", gtsrb_poisoned_test, None)  # See 'BackdoorTrigger.py' for details
    else:
        print("Skipping backdoor pattern application for GTSRB TESTING set (assuming dataset previously poisoned)")

    print()
    # BREAKPOINT 2: Application of Backdoor Trigger
    # breakpoint()

    # TODO (1.X) REDUCING THE SIZE OF DATASETS BY 1/10 CONSIDERING RESOURCE AND TIME CONSTRAINTS

    reduct_ratio = 0.1

    num_cifar_clean_train = int(len(cifar_clean_train) * reduct_ratio)
    num_cifar_clean_erase = len(cifar_clean_train) - num_cifar_clean_train
    cifar_clean_train, _ = torch.utils.data.random_split(
        cifar_clean_train, [num_cifar_clean_train, num_cifar_clean_erase])
    print(f"New reduced number of cifar clean train samples:", len(cifar_clean_train))

    num_cifar_clean_test = int(len(cifar_clean_test) * reduct_ratio)
    num_cifar_clean_erase = len(cifar_clean_test) - num_cifar_clean_test
    cifar_clean_test, _ = torch.utils.data.random_split(
        cifar_clean_test, [num_cifar_clean_test, num_cifar_clean_erase])
    print(f"New reduced number of cifar clean test samples:", len(cifar_clean_test))

    num_gtsrb_clean_train = int(len(gtsrb_clean_train) * reduct_ratio)
    num_gtsrb_clean_erase = len(gtsrb_clean_train) - num_gtsrb_clean_train
    gtsrb_clean_train, _ = torch.utils.data.random_split(
        gtsrb_clean_train, [num_gtsrb_clean_train, num_gtsrb_clean_erase])
    print(f"New reduced number of gtsrb clean train samples:", len(gtsrb_clean_train))

    num_gtsrb_clean_test = int(len(gtsrb_clean_test) * reduct_ratio)
    num_gtsrb_clean_erase = len(gtsrb_clean_test) - num_gtsrb_clean_test
    gtsrb_clean_test, _ = torch.utils.data.random_split(
        gtsrb_clean_test, [num_gtsrb_clean_test, num_gtsrb_clean_erase])
    print(f"New reduced number of gtsrb clean test samples:", len(gtsrb_clean_test))

    print()
    # TODO (1.3) Allocating and transferring 20% of each training set to construct validation sets

    # We have decided to have 80%-20% balance between training and validation sets for our experimentation
    train_ratio = 0.8

    # Splitting clean CIFAR-10 training dataset
    num_cifar_clean_train = int(len(cifar_clean_train) * train_ratio)
    num_cifar_clean_valid = len(cifar_clean_train) - num_cifar_clean_train
    cifar_clean_train, cifar_clean_valid = torch.utils.data.random_split(
        cifar_clean_train, [num_cifar_clean_train, num_cifar_clean_valid])
    print("cifar_clean_train has been split into: cifar_clean_train (80%) & cifar_clean_valid (20%)")

    # Splitting poisoned CIFAR-10 training dataset
    num_cifar_poisoned_train = int(len(cifar_poisoned_train) * train_ratio)
    num_cifar_poisoned_valid = len(cifar_poisoned_train) - num_cifar_poisoned_train
    cifar_poisoned_train, cifar_poisoned_valid = torch.utils.data.random_split(
        cifar_poisoned_train, [num_cifar_poisoned_train, num_cifar_poisoned_valid])
    print("cifar_poisoned_train has been split into: cifar_poisoned_train (80%) & cifar_poisoned_valid (20%)")

    # Splitting clean GTSRB training dataset
    num_gtsrb_clean_train = int(len(gtsrb_clean_train) * train_ratio)
    num_gtsrb_clean_valid = len(gtsrb_clean_train) - num_gtsrb_clean_train
    gtsrb_clean_train, gtsrb_clean_valid = torch.utils.data.random_split(
        gtsrb_clean_train, [num_gtsrb_clean_train, num_gtsrb_clean_valid])
    print("gtsrb_clean_train has been split into: gtsrb_clean_train (80%) & gtsrb_clean_valid (20%)")

    # Splitting poisoned GTSRB training dataset
    num_gtsrb_poisoned_train = int(len(gtsrb_poisoned_train) * train_ratio)
    num_gtsrb_poisoned_valid = len(gtsrb_poisoned_train) - num_gtsrb_poisoned_train
    gtsrb_poisoned_train, gtsrb_poisoned_valid = torch.utils.data.random_split(
        gtsrb_poisoned_train, [num_gtsrb_poisoned_train, num_gtsrb_poisoned_valid])
    print("gtsrb_poisoned_train has been split into: gtsrb_poisoned_train (80%) & gtsrb_poisoned_valid (20%)")

    return (
        cifar_clean_train, cifar_clean_valid, cifar_clean_test,
        cifar_poisoned_train, cifar_poisoned_valid, cifar_poisoned_test,
        gtsrb_clean_train, gtsrb_clean_valid, gtsrb_clean_test,
        gtsrb_poisoned_train, gtsrb_poisoned_valid, gtsrb_poisoned_test
    )