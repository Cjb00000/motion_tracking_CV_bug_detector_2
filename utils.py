from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform
def get_model(num_classes, device, weights_path=None):
    model = models.resnet18(pretrained=True)  # loads in the architecture for ResNet18 (pretrained=True uses pretrained weights (the numbers and matrices used in the model))

    # Replace the final classification layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # output units for classification
    model = model.to(device)

    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for binary classification (loss == error between prediction and truth)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Adjust optimizer as needed
    return model, criterion, optimizer

def get_data_from_dir(data_path):
    data = []
    class_names = os.listdir(data_path) #returns a list of our folders in dataset_dir
    for i, class_name in enumerate(class_names): #this will iterate the number of times you have class_names (number of directories in dataset_dir)
        class_dir = os.path.join(data_path, class_name) #joins the strings with a \ of dataset_dir and class_name so it can get the correct path
        image_files = os.listdir(class_dir) #list all the images in the directory of class_dir
        label = i  # Assign a unique label for each class
        #print(image_files)
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            data.append((image_path, label))
    return data, class_names

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig('loss.png')
    return