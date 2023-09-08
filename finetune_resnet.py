import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_transforms, get_model, get_data_from_dir, plot_loss
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU
class MulticlassDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # List of (image_path, label) pairs
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':

    batch_size = 32  # Adjust the batch size as needed
    shuffle = True    # Shuffle the data during training
    num_workers = 4  # Number of CPU cores for data loading (adjust as needed)
    task_name = 'moth'
    train_dir = f'./{task_name}_task/training_data/chips'  # Update with the path to your dataset directory
    val_dir = f'./{task_name}_task/validation_data/chips'

    train_data, class_names = get_data_from_dir(train_dir)
    val_data, class_names = get_data_from_dir(val_dir)

    transform = get_transforms()

    # Assuming 'data' is a list of (image_path, label) pairs
    train_dataset = MulticlassDataset(train_data, transform=transform)
    val_dataset = MulticlassDataset(val_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # used for efficiently loading and batching data during the training of machine learning models, particularly deep learning models
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model, criterion, optimizer = get_model(len(class_names), device)
    # Assuming you have a DataLoader named 'train_loader' for your training dataset
    num_epochs = 32  # Adjust the number of epochs as needed #epoch == passing a single image through your model changes the weights of the model after your model has seen every data point
    #does back propagation on each data point, which uses loss to determine how much it should change the weights of the model so that the next prediction is better. This process is one epoch
    #weights/parameters are the numbers in the matrices in the model (these are being changed)
    #multiple epochs so it learns from the same data multiple times

    train_minimum_loss = 10e9
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        val_running_loss = 0.0

        for inputs, labels in train_loader:
            model.train()  # Set the model to training mode
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradient buffers
            #If you don't zero out the gradients at the beginning of each batch, the gradients will accumulate across multiple batches (accumulate slopes)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_running_loss += loss.item()

        for inputs, labels in val_loader:
            model.eval()  # Set the model to training mode
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss

            val_running_loss += loss.item()

        train_epoch_loss = train_running_loss / len(train_loader) #finds the average loss per epoch
        val_epoch_loss = val_running_loss / len(val_loader)
        if train_epoch_loss < train_minimum_loss:
            torch.save(model.state_dict(), f'{task_name}_task_{len(class_names)}-class_resnet18.pth')
            train_minimum_loss = train_epoch_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_epoch_loss:.4f} Val Loss: {val_epoch_loss:.4f}")
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
    print("Training complete")
    plot_loss(train_losses, val_losses)

