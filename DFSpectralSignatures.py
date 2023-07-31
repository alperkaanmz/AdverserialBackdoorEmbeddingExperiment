
# TODO Algorithm for Dataset Filtering utilizing Spectral Signatures

import numpy as np

# Define the threshold for identifying potential backdoor triggers
threshold = 0.1  # Adjust according to your needs

# Dataset Filtering using spectral signatures
def apply(model, train_dataset, threshold):
    model.eval()  # Switch to evaluation mode

    # Obtain the latent representations (activations) for clean inputs
    clean_latent_representations = []

    for inputs, labels in train_dataset:
        outputs = model(inputs)
        clean_latent_representations.append(outputs.detach().numpy())

    clean_latent_representations = np.concatenate(clean_latent_representations)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(clean_latent_representations, rowvar=False)

    # Perform Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(covariance_matrix)

    # Compute the outlier score for each input
    outlier_scores = []

    for inputs, labels in train_dataset:
        outputs = model(inputs)
        latent_representation = outputs.detach().numpy()
        residual = latent_representation - np.mean(clean_latent_representations, axis=0)
        outlier_score = np.sqrt(np.sum((residual @ np.linalg.inv(np.diag(S)) * residual), axis=1))
        outlier_scores.append(outlier_score)

    # Find the indices of inputs with outlier scores above the threshold
    backdoor_trigger_indices = np.where(outlier_scores > threshold)[0]

    # Print the number of potential backdoor triggers
    print(f"Number of potential backdoor triggers = {len(backdoor_trigger_indices)}")

    # Remove poisoned inputs from the training set
    filtered_train_dataset = [data for i, data in enumerate(train_dataset) if i not in backdoor_trigger_indices]

    return filtered_train_dataset, len(filtered_train_dataset), len(backdoor_trigger_indices)