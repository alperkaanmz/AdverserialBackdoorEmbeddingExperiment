
import Models
import torch
import torchvision.models as models
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
import RegularTraining

# TODO Dataset Filtering by Activation Clustering (including Exclusionary Reclassification)

def apply(model, inputs):

    # Obtain the latent representations of the inputs.
    activations = model(inputs)

    # Reduce the dimensionality of the latent representations.
    ica = FastICA(n_components=10)
    latent_representations = ica.fit_transform(activations)

    # Cluster the latent representations using k-means clustering.
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(latent_representations)

    # Identify the poisoned cluster.
    poisoned_cluster = clusters == 1

    # Return a boolean mask indicating which inputs are in the poisoned cluster.
    return poisoned_cluster


def exclusionary_reclassification(model, poisoned_cluster, train_dataset, train_batch_size, valid_dataset,
                                  valid_batch_size, loss_function, optim_algo, num_epochs):
    # Declaring num_classes
    num_classes_cifar = 10
    num_classes_gtsrb = 43

    # Creating a new model with the same architecture as the old model
    if isinstance(model, models.densenet.DenseNet):
      new_model = Models.DenseNetBC(num_classes_gtsrb, L=100, k=12)
    elif isinstance(model, models.vgg.VGG):
      new_model = Models.VGG(num_classes_cifar)
    else:
      raise ValueError("Unknown model architecture.")

    # Training the new model on the clean inputs
    RegularTraining.apply(new_model, train_dataset, train_batch_size, valid_dataset, valid_batch_size, loss_function,
                          optim_algo, num_epochs)

    # Setting the newly trained model to evaluation mode
    new_model.eval()

    # Extracting input and label subsets of the dataset previously used for training
    train_inputs = train_dataset[:][0]  # extracting the input set
    train_labels = train_dataset[:][1]  # extracting the label set

    # Obtain the predictions on the poisoned inputs
    poisoned_inputs = train_inputs[poisoned_cluster]
    poisoned_predictions = new_model(poisoned_inputs)

    # Calculate poisoned accuracy
    poisoned_labels = train_labels[poisoned_cluster]
    poisoned_accuracy = accuracy(poisoned_predictions, poisoned_labels)

    # Display results
    if poisoned_accuracy > 0.5:
      print('Exclusionary reclassification: The new model has been poisoned.')
    else:
      print('Exclusionary reclassification: The new model has not been poisoned.')

    return new_model

def accuracy(predictions, labels):
    # Calculate the accuracy of predictions
    _, predicted = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()  # TODO invalid implementation, awaiting correction
    accuracy = correct / total
    return accuracy
