import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import json



# CNN WORK (BOGDAN'S PART):

model_path = 'mushroom_classfier1.pth'


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


# Load the model(Once it is finished):
def load_model(model_path):
    model = MushroomClassfier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model




transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    # The below codes that are commented out were used for data augmentation, but I ultimately decided to take them out as it was more useful to keep the dataset normal. --> Kept for reasons of making model potetnially more accurate in future.
    # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    # transforms.RandomRotation(15),      # Rotate images by up to 15 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

def classify(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Make a prediction:
    output = model(image)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    class_name = idx_to_class[class_index]
    return class_name


idx_to_class = {0: 'edible', 1:'poisonous'}
# Load the model into the application:
model = load_model(model_path)


print('Idenityfing mushroom:')
image_path = 'rod_sopp.png'
result = classify(image_path, model, transform)
print(result)

