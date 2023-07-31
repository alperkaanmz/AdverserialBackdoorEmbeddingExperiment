import numpy as np

# TODO Feature Pruning Algorithm to be utilized during defensive evaluations

# Feature Pruning as a detection technique
def apply_feature_pruning(model, train_dataset, num_classes, threshold, full):
    model.eval()  # Switch to evaluation mode

    # Obtain the latent representations (activations) for clean inputs
    clean_latent_representations = []

    for inputs, labels in train_dataset:
        outputs = model(inputs)
        clean_latent_representations.append(outputs.detach().numpy())

    clean_latent_representations = np.concatenate(clean_latent_representations)

    # Iterate over each class
    for class_label in range(num_classes):
        # Obtain the latent representations (activations) for inputs of the specific class
        class_inputs = [inputs for inputs, labels in train_dataset if labels == class_label]
        class_latent_representations = []

        for inputs in class_inputs:
            outputs = model(inputs)
            class_latent_representations.append(outputs.detach().numpy())

        class_latent_representations = np.concatenate(class_latent_representations)

        # Calculate the perturbation for each input
        perturbations = np.linalg.norm(class_latent_representations - clean_latent_representations, axis=1)

        # Find the indices of inputs with perturbations above the threshold
        backdoor_trigger_indices = np.where(perturbations > threshold)[0]

        # Print the number of potential backdoor triggers for the class
        print(f"Class {class_label}: Number of potential backdoor triggers = {len(backdoor_trigger_indices)}")

        if full:
            # Remove all neurons detected as potential backdoor triggers
            for index in backdoor_trigger_indices:
                model = remove_neuron(model, index)
        else:
            # Remove a single neuron with the highest perturbation
            if len(backdoor_trigger_indices) > 0:
                max_perturbation_index = backdoor_trigger_indices[np.argmax(perturbations[backdoor_trigger_indices])]
                model = remove_neuron(model, max_perturbation_index)

    return model

def remove_neuron(model, neuron_index):
    # Get the parameters of the model
    params = list(model.parameters())

    # Find the layer and index of the neuron to be removed
    layer_index = None
    neuron_layer_index = None

    for i, param in enumerate(params):
        if len(param.shape) == 2 and neuron_index < param.shape[1]:
            layer_index = i
            neuron_layer_index = neuron_index
            break
        elif len(param.shape) == 4 and neuron_index < param.shape[0] * param.shape[1]:
            layer_index = i
            neuron_layer_index = neuron_index % (param.shape[0] * param.shape[1])
            break

    if layer_index is not None and neuron_layer_index is not None:
        # Remove the neuron from the layer
        if len(params[layer_index].shape) == 2:
            params[layer_index] = np.delete(params[layer_index],
                                            neuron_layer_index, axis=1)
        elif len(params[layer_index].shape) == 4:
            params[layer_index] = np.delete(params[layer_index],
                                            neuron_layer_index // params[layer_index].shape[1], axis=0)

        # Reconstruct the model with the modified parameters
        new_model = type(model)(*params)

        # Copy over the remaining model state
        new_model.load_state_dict(model.state_dict())

        return new_model

    return model
