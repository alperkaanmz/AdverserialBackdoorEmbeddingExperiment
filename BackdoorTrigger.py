
# TODO Backdoor Trigger / Embedding (Generalized Python Function for Applying Backdoor Triggers)

'''
-Specific class or target label to be associated with the backdoor trigger
-Specific trigger pattern or image that will activate the backdoor
-Poison the training dataset by injecting the trigger pattern into a subset of the images from the chosen target class
-Ensure that the poisoned images are labeled correctly with the target label

A selected portion of training data will be separated and 'poisoned'
    by adding a white 4x4 square trigger pattern to
    the bottom right corner of 5% of training samples,
    which will have their target label set to yt = 2
'''

import torch
import random

# TODO IMPLEMENTING THE PYTHON FUNCTION THAT APPLIES BACKDOOR TRIGGER ONTO A DATASET WITH PAPER SPECIFIC CHARACTERISTICS
def apply_cifar(purpose, dataset, target_label):  # target label will be set to '2' in 'Main'
    # Declaring our 'Backdoor Pattern' in accordance to given dataset type:
    trigger_pattern = torch.ones((4, 4, 3)) * 255  # 4x4 white square backdoor pattern
    num_samples_to_modify = int(len(dataset) * 0.05)  # to be applied on 5% of all training samples

    # Randomly picking training samples (corresponding to 5% of given datasets training)
    samples_to_modify = random.sample(range(len(dataset)), num_samples_to_modify)

    # Applying our backdoor trigger onto the selected subset of training samples
    for idx in samples_to_modify:
        # Applying backdoor trigger to bottom right corner
        image = dataset.data[idx]
        image[-4:, -4:] = trigger_pattern

        # Apply change in target label only if the purpose is for 'train' and a target label is specified
        if purpose == "train" and target_label is not None:
            dataset.targets[idx] = target_label

    print("Backdoor pattern applied on 5% of", type(dataset).__name__, purpose, "dataset successfully!")
    print("(pattern applied on a total of", num_samples_to_modify, "samples out of total", len(dataset), "available)")

def apply_gtsrb(purpose, dataset, target_label):
    # Assuming images are in [height, width, channels] format
    trigger_pattern = torch.ones((3, 4, 4)) * 255
    num_samples_to_modify = int(len(dataset) * 0.05)  # Modify 5% of all training samples
    samples_to_modify = random.sample(range(len(dataset)), num_samples_to_modify)

    # Initializing a new list for appending poisoned dataset
    poisoned_dataset = []

    # Iterating over dataset, poisoning samples of selected indices
    for idx in range(len(dataset)):
        image, label = dataset[idx]

        # Apply backdoor trigger on sample if selected for poisoning
        if idx in samples_to_modify:
            image[:, -4:, -4:] = trigger_pattern  # Apply trigger to bottom right corner
            # Change target label if this backdoor pattern application is purposed for a traning sample
            # (which also requires the target label to be updated to an arbitrary choice of attackers)
            if purpose == "train" and target_label is not None:
                label = target_label
        # Append image/label pair back into dataset
        poisoned_dataset.append((image, label))

    # Insert the substitution
    dataset.samples = poisoned_dataset

    print(f"Backdoor pattern applied on 5% of {type(dataset).__name__} {purpose} dataset successfully!")
    print(f"(pattern applied on a total of {num_samples_to_modify} samples out of total {len(dataset)} available)")