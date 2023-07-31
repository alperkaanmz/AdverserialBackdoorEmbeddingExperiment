
# TODO Regular Model Training (standard, without adversarial embedding additions)

import torch
import torch.utils.data
import Validation

def apply(
        model, train_dataset, train_batch_size, valid_dataset, valid_batch_size, loss_function, optim_algo, num_epochs
):
    # Applying cuda optimizers for faster training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Declaring 'Criterion', representing 'Loss Function' to be utilized during the training process
    criterion = loss_function
    # Declaring 'Optimizer', representing 'Optimization Algorithm' to be used for minimizing the loss during training
    optimizer = optim_algo

    # TODO DATA LOADERS FOR TRAINING
    print("Initializing train_dataset DataLoader...")
    train_batch_size = train_batch_size  # Declaring the batch size for training
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    print("train_dataset DataLoader initialized")

    # Declaring data to be collected
    list_train_loss = []
    list_train_accu = []
    list_valid_loss = []
    list_valid_accu = []
    latent_representations = []  # TODO in an ideal case, we should have been able to extract this from training as well

    # Implementing the training loop
    for epoch in range(num_epochs):
        model.train()  # Switching our model of interest in 'Training Mode' TODO MODEL SET TO TRAINING MODE
        # Initiating a counter, for tracking the accumulated loss within each epoch, resets to 0.0 between epochs
        running_loss = 0.0  # TODO TRACK LOSS PER EPOCH
        running_corr = 0  # TODO TRACK CORRECT PREDICTIONS PER EPOCH
        elapsed = 0  # TODO TRACK NUMBER OF PREDICTIONS MADE UNTIL NOW

        # Initiating the 'Iteration'
        for inputs, labels in train_data_loader:  # TODO ITERATE BATCHES
            optimizer.zero_grad()  # clearing gradients of all model params, preventing accum. from previous iterations
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # inputs passed through model to obtain predicted outputs TODO PREDICTION & LOSS
            loss = criterion(outputs, labels)  # loss between predicted output and actual label calculated via criterion
            loss.backward()  # computing gradients of models parameters with respect to loss TODO BACKPROPAGATION
            optimizer.step()  # updating models parameters based on our chosen optimization algorithm TODO OPTIMIZATION
            # Accumulating loss for each batch, multiplying loss value with batch size, added to 'Running Loss'
            running_loss += loss.item() * inputs.size(0)  # TODO INCREMENT RUNNING LOSS
            # Accumulating correct predictions for each batch, will be added onto 'Running Corr'
            label, pred = torch.max(outputs, 1)
            running_corr += torch.sum(pred == labels.data).item()

            # Checking the speed of our training process on a single line indicator
            elapsed += 1
            # print(f"Batch: [{elapsed}/{math.ceil(len(train_dataset)/inputs.size(0))}]")  TODO REACTIVATE IF USEFUL

        # TODO PRINTING CONTEXTUALLY RELEVANT INFORMATION EMERGING FROM EACH EPOCH OF TRAINING
        epoch_loss = running_loss / len(train_dataset)
        epoch_accu = running_corr / len(train_dataset)
        print(model.name, f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accu:.4f}')
        valid_loss, valid_accu = Validation.apply(model, valid_dataset, valid_batch_size, loss_function)

        # Appending relevant data to be returned as a result of this training run
        list_train_loss.append(epoch_loss)
        list_train_accu.append(epoch_accu)
        list_valid_loss.append(valid_loss)
        list_valid_accu.append(valid_accu)
        # latent_representations would be appended here

        '''
        Considering this training code block performing regular training procedure onto given model/dataset pairs,
        we should normally be extracting relevant latent representations in order to be processed by our implemented
        defensive algorithm functionalities, which we couldn't wrap our heads around if you are reading this comment
        '''
        return list_train_loss, list_train_accu, list_valid_loss, list_valid_accu