import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import math
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


# This code file will be used to train the model, and load it into an actual model to use in the plant_classifier file:
print('Transforming Data...\n')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor()
    # transforms.Resize((224, 224)),  
    # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    # transforms.RandomRotation(15),      # Rotate images by up to 15 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.ToTensor(),
])

print('Loading Dataset...\n')
# Load the dataset:
dataset = datasets.ImageFolder(root='mushroom_database', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('Creating Neural Network...\n')
class MushroomClassfier(nn.Module):
    def __init__(self):
        super(MushroomClassfier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


print('Instantiating Model, Loss Function, and Optimizer...\n')
# Instantiate the model, and define loss function and optimizer:
model = MushroomClassfier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print('Training Model...\n')
# Train the model using the Mushroom_classifier Neural Network: 
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    CCR = 100 * test_correct/test_total # Get the Classfier Correct Rate :)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss: 4f}, CCR: ~{CCR: .2f}%\n')


print('Finished Training. Saving Model...\n')
# Save the model:
torch.save(model.state_dict(), 'mushroom_classfier1.pth')


print('Model Saved. Thank you!')
sys.exit(0) # Exit 0 on success. 
 
