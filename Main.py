import os
import torch
import Datasets
# import BackdoorTrigger (already utilized inside Datasets procedure)
import Models
import RegularTraining
# import BasicEvaluation (already utilized inside ModelTraining for validation purposes)
import torch.nn as nn
import torch
import torchvision.models as models

'''
    For the upcoming experimentation phase, we have the following general roadmap:
        0) Training of initial models on clean data, producing 'healthy' model instances
        1) Naive poisoning of healthy models, producing 'naived' model instances
        2) Application of proposed defence mechanisms on naively poisoned models
            which should theoretically prove naive poisoning ineffective against proposed defenses
        3) Training of naived models on poisoned dataset with adversarial embedding (producing 'adversed' models)
            this phase could be phrased as "reacting to model's defences"
        4) Application of proposed defence mechanisms on adversarial trained models
            which should theoretically prove adversarial embedding effective against proposed static defenses
    '''

def main():

    # Initializing and preparing our datasets (CIFAR-10 and GTSRB) to be utilized
    apply_trigger = False  # this is our own flag to prevent re-poisoning of previously poisoned datasets
    # (backdoor pattern application currently turned because previously poisoned data had been already prepared)
    healthy_models_present = False
    naived_models_present = False
    adversed_models_present = False

    # TODO Initializing datasets to be utilized during our experimentation
    (cifar_clean_train, cifar_clean_valid, cifar_clean_test,
        cifar_poisoned_train, cifar_poisoned_valid, cifar_poisoned_test,
        gtsrb_clean_train, gtsrb_clean_valid, gtsrb_clean_test,
        gtsrb_poisoned_train, gtsrb_poisoned_valid, gtsrb_poisoned_test) = Datasets.prepare(apply_trigger)

    num_classes_cifar = 10
    print("Number of CIFAR-10 unique classes:", num_classes_cifar)
    num_classes_gtsrb = 43
    print("Number of GTSRB unique classes:", num_classes_gtsrb)

    # TODO Initializing our models, firstly to be trained on clean datasets
    model_densenetbc_gtsrb = None
    print("Reference for initial DenseNet-BC model instance declared")
    model_vgg_cifar = None
    print("Reference for initial VGG model instance declared")

    # TODO Checking whether we already have previously trained models in our storage directory ready for use
    # Path were any previously saved DenseNetBC model should be saved at
    densenetbc_path = None
    if adversed_models_present:
        densenetbc_path = '.models/adversed/densenetbc.pth'
    elif naived_models_present:
        densenetbc_path = '.models/naived/densenetbc.pth'
    elif healthy_models_present:
        densenetbc_path = '.models/healthy/densenetbc.pth'
    # Path were any previously saved VGG model should be saved at
    vgg_path = None
    if adversed_models_present:
        vgg_path = '.models/adversed/vgg.pth'
    elif naived_models_present:
        vgg_path = '.models/naived/vgg.pth'
    elif healthy_models_present:
        vgg_path = '.models/healthy/vgg.pth'

    print()
    print("Checking if previous DenseNet-BC model instance present:")
    prev_densenetbc = True
    try:
        # Attempt to load the model
        model_densenetbc_gtsrb = torch.load(densenetbc_path)
        print("Previous DenseNetBC model loaded.")
        # prev_densenetbc = True
    except FileNotFoundError:
        print("No previous DenseNetBC model found.")
        prev_densenetbc = False
    except Exception as e:
        print("An error occurred while loading DenseNet-BC:", str(e))
        prev_densenetbc = False

    print()
    print("Checking if previous VGG model instance present:")
    prev_vgg = True
    try:
        # Attempt to load the model
        model_vgg_cifar = torch.load(vgg_path)
        print("Previous VGG model loaded.")
        # prev_vgg = True
    except FileNotFoundError:
        print("No previous VGG model found.")
        prev_vgg = False
    except Exception as e:
        print("An error occurred while loading VGG:", str(e))
        prev_vgg = False

    print()

    # Evaluation TABLE 1: DenseNet will be trained on GTSRB (also including 'paper' declared layer amount L = 100)
    if not prev_densenetbc:
        model_densenetbc_gtsrb = Models.DenseNetBC(num_classes_gtsrb, L=100, k=12)
        print("Healthy DenseNet-BC (densenet121) instance created: num_classes = ", num_classes_gtsrb, "L=100", "k=12")

    # Evaluation TABLE 1: VGG will be trained on CIFAR-10
    if not prev_vgg:
        model_vgg_cifar = Models.VGG(num_classes_cifar)  # VGG instance now has 10 classifications
        print("Healthy VGG (vgg19) instance created: num_classes = ", num_classes_cifar)

    print()

    # -----------------------------------------------------------------------------------------------------------------
    # TODO OUR DATASETS AND MODELS ARE NOW READY FOR EXPERIMENTATION SCENARIOS
    # -----------------------------------------------------------------------------------------------------------------
    # TODO FROM THIS POINT ON, VARIOUS EVALUATION AND TESTING SCENARIOS WILL BE SCRIPTED IN THIS PYTHON FILE
    # -----------------------------------------------------------------------------------------------------------------
    # TODO {SCENARIO 0}: Training both models (DenseNet-BC with GTSRB, VGG with CIFAR-10) on clean datasets
    #   in order to measure their performances in standard and healthy conditions

    batchsize_train = 64
    print(f"Assigning batch_size_train = {batchsize_train} considering the balance between datasets")

    batchsize_valid = 16
    print(f"Assigning batchsize_valid = {batchsize_valid} considering the balance between datasets")

    batchsize_test = 16
    print(f"Assigning batchsize_test = {batchsize_test} considering the balance between datasets")

    # Evaluation TABLE 1: our Tuning LR is 10^-1 for both models
    tuninglr_training_healthy = pow(10, -1)
    print("(for training of healthy models) TuningLR =", tuninglr_training_healthy)

    # Evaluation TABLE 1: our Regularization Parameter 'lambda' is 1 for both models
    lambda_training_healthy = 1
    print("Regularization Parameter 'lambda' =", lambda_training_healthy)

    # We have chosen 'Cross Entrophy' as our loss function for our training procedures
    lossfunction_training = torch.nn.CrossEntropyLoss()  # TODO CHANGE THIS IF REQUIRED LATER ON
    print("Cross Entropy Loss-Function initiated")

    # Declaring Adam for DenseNet-BC model (also assuming momentum = 0.9)
    optimalgo_densenetbc_gtsrb_healthy = torch.optim.Adam(
        model_densenetbc_gtsrb.parameters(),
        lr=tuninglr_training_healthy,
        weight_decay=lambda_training_healthy
    )
    print("Optimization algorithm Adam initiated for healthy DenseNet-BC")

    # Declaring SGD for VGG model (also assuming momentum = 0.9)
    optimalgo_vgg_cifar_healthy = torch.optim.Adam(
        model_vgg_cifar.parameters(),
        lr=tuninglr_training_healthy,
        weight_decay=lambda_training_healthy
    )
    print("Optimization algorithm Adam initiated for healthy VGG")

    # Evaluation TABLE 1: both base models will be trained for 30 Epochs
    epochs_training_healthy = 30
    print("Epochs for 'initial healthy training' set as '30' ")

    # TODO {Scenario 0: Training on clean data}

    if not healthy_models_present:  # prevents unnecessary re-training of previously trained DenseNet-BC
        print("Applying standard, healthy training on DenseNet-BC:")
        RegularTraining.apply(
            model_densenetbc_gtsrb,  # DenseNet-BC
            gtsrb_clean_train,  # clean GTSRB Dataset allocated for training
            batchsize_train,
            gtsrb_clean_valid,  # clean GTSRB Dataset allocated for validation
            batchsize_valid,
            lossfunction_training,  # CrossEntropyLoss
            optimalgo_densenetbc_gtsrb_healthy,  # SGD
            epochs_training_healthy  # 30
            )
        densenetbc_save_dir = 'models/healthy'
        os.makedirs(densenetbc_save_dir, exist_ok=True)
        model_path = os.path.join(densenetbc_save_dir, 'densenetbc.pth')
        torch.save(model_densenetbc_gtsrb.state_dict(), model_path)

    if not healthy_models_present:  # prevents unnecessary re-training of previous trained VGG model
        print("Applying standard, healthy training on VGG:")
        RegularTraining.apply(
            model_vgg_cifar,  # VGG
            cifar_clean_train,  # clean CIFAR-10 Dataset allocated for training
            batchsize_train,
            cifar_clean_valid,  # clean CIFAR-10 Dataset allocated for validation
            batchsize_valid,
            lossfunction_training,  # CrossEntropyLoss
            optimalgo_vgg_cifar_healthy,  # SGD
            epochs_training_healthy  # 30
        )
        vgg_save_dir = 'models/healthy'
        os.makedirs(vgg_save_dir, exist_ok=True)
        model_path = os.path.join(vgg_save_dir, 'vgg.pth')
        torch.save(model_vgg_cifar.state_dict(), model_path)

    # TODO {End of scenario 0}

    # We now have a 'healthy model' instance ready for both our architectures

    # TODO {Scenario 1: training on poisoned data, naive poisoning of models}
    if not naived_models_present:
        print("Applying naive poisoning training on DenseNet-BC with poisoned GTSRB dataset:")
        RegularTraining.apply(
            model_densenetbc_gtsrb,  # DenseNet-BC
            gtsrb_poisoned_train,  # poisoned GTSRB Dataset allocated for training
            batchsize_train,
            gtsrb_poisoned_valid,  # poisoned GTSRB Dataset allocated for validation
            batchsize_valid,
            lossfunction_training,  # whichever 'Loss Function' decided for 'this' training scenario
            optimalgo_densenetbc_gtsrb_healthy,  # whichever 'Optimization Algorithm' decided for 'this' training scenario
            epochs_training_healthy  # number of epochs decided for 'this' training scenario
        )
        densenetbc_save_dir = 'models/naived'
        os.makedirs(densenetbc_save_dir, exist_ok=True)
        model_path = os.path.join(densenetbc_save_dir, 'densenetbc.pth')
        torch.save(model_densenetbc_gtsrb.state_dict(), model_path)

    if not naived_models_present:
        print("Applying naive poisoning training on VGG with poisoned CIFAR-10 dataset:")
        RegularTraining.apply(
            model_vgg_cifar,  # VGG
            cifar_poisoned_train,  # poisoned CIFAR Dataset allocated for training
            batchsize_train,
            cifar_poisoned_valid,  # poisoned CIFAR Dataset allocated for validation
            batchsize_valid,
            lossfunction_training,  # whichever 'Loss Function' decided for 'this' training scenario
            optimalgo_vgg_cifar_healthy,  # whichever 'Optimization Algorithm' decided for 'this' training scenario
            epochs_training_healthy  # number of epochs decided for 'this' training scenario
        )
        vgg_save_dir = 'models/naived'
        os.makedirs(vgg_save_dir, exist_ok=True)
        model_path = os.path.join(vgg_save_dir, 'vgg.pth')
        torch.save(model_vgg_cifar.state_dict(), model_path)

    # TODO {End of scenario 1}

    # TODO {Scenario 2: Application of defenses on naively poisoned models}
    # ((naived vgg on cifar-10)(naived densenetbc on gtsrb))
    # <- (FeaturePruning, DFSpectralSigranutes, DFActivationClustering)
    # TODO FEATURE PRUNING, DF SPECTRAL SIGNATURES vs DF ACTIVATION CLUSTERIGN

    # TODO {End of Scenario 2}

    # TODO {Scenario 3: training naively poisoned models with adversarial embedding}

    '''
    We are to insert the following Discriminator Architecture into our models for our adversarial attack:
        Discriminator Layers:
            Fully-connected, 256 channels, Leaky ReLU & Batchnorm activation
            Fully-connected, 128 channels, Leaky ReLU & Batchnorm activation
            Fully-connected, 1 channel, Softmax
    '''
    # TODO instantiating our discriminators to be inserted into our models during feature pruning defense procedures:

    discrimination_learning_rate = pow(10, -3)

    discriminator_densenetbc_gtsrb = nn.Sequential(
        nn.Linear(256, 128),  # first FC layer into Second FC layer
        nn.LeakyReLU(),  # leaky ReLU activation in first FC
        nn.BatchNorm1d(128),  # BatchNorm activation in first FC
        nn.Linear(128, 1),
        nn.Softmax(dim=1)
    )
    # initiating the optimizer for our discriminator with given discrimination learning rate applied
    discrim_optimalgo_densenet_gtsrb = torch.optim.Adam(discriminator_densenetbc_gtsrb.parameters(),
                                                        lr=discrimination_learning_rate)

    discriminator_vgg_cifar = nn.Sequential(
        nn.Linear(256, 128),  # first FC layer into Second FC layer
        nn.LeakyReLU(),  # leaky ReLU activation in first FC
        nn.BatchNorm1d(128),  # BatchNorm activation in first FC
        nn.Linear(128, 1),
        nn.Softmax(dim=1)
    )
    # initiating the optimizer for our discriminator with given discrimination learning rate applied
    discrim_optimalgo_vgg_cifar = torch.optim.Adam(discriminator_vgg_cifar.parameters(),
                                                   lr=discrimination_learning_rate)

    '''
    For simulating the upcoming scenarios based on the evaluation of defenses against out adversarial embedding attack:
        We would require the following training combinations of models and (poisoned datasets):
        [According to given TABLE 2, addressing the adversarial training combinations performed during experimentation]
            1) DenseNet-BC on poisoned CIFAR-10
                Epochs: 30
                Tuning LR: 10^-4
                Discrimination LR = 10^-3
                lambda = 50
            2) DenseNet-BC on poisoned GTSRB
                Epochs = 30
                Tuning LR: 10^-4
                Discrimination LR = 10^-3
                lambda = 20
            3) VGG on poisoned CIFAR-10
                Epochs: 1000
                Tuning LR: 10^-3
                Discrimination LR = 10^-3
                lambda = 20
            4) VGG on poisoned GTSRB
                Epochs: 30
                Tuning LR = 10^-4
                Discrimination LR = 10^-3
                lambda = 20
    '''

    # TODO We will continue with the previously naively poisoned DenseNet-BC on GTSRB and VGG on CIFAR-10

    # Evaluation TABLE 2: our Tuning LR is the following for both models:
    tuninglr_training_adversary_densenetbc = pow(10, -1-4)
    tuninglr_training_adversary_vgg = pow(10, -3)
    print("TuningLR for adversary DenseNet-BC =", tuninglr_training_adversary_densenetbc)
    print("TuningLR for adversary VGG =", tuninglr_training_adversary_vgg)

    # Evaluation TABLE 2: our Regularization Parameter 'lambda' is 20 for both models relevant models
    lambda_training_adversary = 20
    print("Regularization Parameter 'lambda' =", lambda_training_adversary)

    # Declaring Adam for DenseNet-BC model (also assuming momentum = 0.9)
    optimalgo_densenetbc_gtsrb_adversary = torch.optim.Adam(
        model_densenetbc_gtsrb.parameters(),
        lr=tuninglr_training_adversary_densenetbc,
        weight_decay=lambda_training_adversary
    )
    print("Optimization algorithm Adam initiated for healthy DenseNet-BC")

    # Declaring Adam for VGG model (also assuming momentum = 0.9)
    optimalgo_vgg_cifar_adversary = torch.optim.Adam(
        model_vgg_cifar.parameters(),
        lr=tuninglr_training_adversary_vgg,
        weight_decay=lambda_training_adversary
    )
    print("Optimization algorithm Adam initiated for healthy VGG")

    # Evaluation TABLE 2: training epochs for adversarial training given as:
    epochs_adversary_densenetbc_gtsrb = 30
    epochs_adversary_vgg_cifar = 1000

    # TODO Inserting discriminator network into DenseNet-BC model instance:
    num_gtsrb_features = model_densenetbc_gtsrb.fc.in_features
    # Replacing the last fully connected layer of network with the discriminator
    model_densenetbc_gtsrb.fc = discriminator_densenetbc_gtsrb
    # Adjusting input size of discriminator
    discriminator_densenetbc_gtsrb.input_size = num_gtsrb_features
    # Freezing parameters of the pre-trained layers
    for param in model_densenetbc_gtsrb.parameters():
        param.requires_grad = False

    # TODO Inserting discriminator network into VGG model instance:
    num_vgg_features = model_vgg_cifar.classifier[-1].in_features
    # Replacing the last fully connected layer of network with the discriminator
    model_vgg_cifar.classifier[-1] = discriminator_vgg_cifar
    # Freezing parameters of the pre-trained layers
    for param in model_vgg_cifar.features.parameters():
        param.requires_grad = False

    # TODO ADVERSARIAL EMBEDDING / ADVERSARIAL TRAINING

    # TODO {End of Scenario 3}

    # TODO {Scenario 4: Application of defenses on adversarially embedded models}
    # ((naived vgg on cifar-10)(naived densenetbc on gtsrb))
    # <- (FeaturePruning, DFSpectralSigranutes, DFActivationClustering)

    # FEATURE PRUNING, DF SPECTRAL SIGNATURES vs DF ACTIVATION CLUSTERIGN

    # TODO {End of Scenario 4}

if __name__ == "__main__":
    main()