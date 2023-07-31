import torch
import torch.utils.data

# TODO Model Testing (utilized for both testing scenarios and model training validation phase)
# Our method intakes 'Model' and 'Dataset', used for both training validation and scenario testing
def apply(model, dataset, batch_size, loss_function):
    # Declaring 'Criterion', representing 'Loss Function' to be utilized during the testing process
    criterion = loss_function

    # TODO DATA LOADER FOR VALIDATION
    batch_size = batch_size  # Declaring the batch size for testing
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()  # Switching the model to 'Evaluation Mode'

    running_loss = 0.0  # TODO TRACK LOSS DURING VALIDATION
    running_corr = 0  # TODO TRACK CORRECT PREDICTIONS DURING VALIDATION

    with torch.no_grad():  # Disabling gradient calculation during validation
        for inputs, labels in data_loader:
            outputs = model(inputs)  # Forward pass to obtain predicted outputs
            loss = criterion(outputs, labels)  # Calculate loss between predicted output and actual label
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss

            # Calculate number of correct predictions
            _, preds = torch.max(outputs, 1)
            running_corr += torch.sum(preds == labels.data).item()

    # Calculate average loss and accuracy
    loss = running_loss / len(dataset)
    accu = running_corr / len(dataset)

    # Printing results
    print(f'Validation: Loss: {loss:.4f}, Accuracy: {accu:.4f}')

    return loss, accu