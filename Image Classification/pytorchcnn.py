#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import imghdr
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch.nn.functional as F


# In[2]:


# Define the Dataset class
DataDir = 'image data'
ImageEx = ['jpeg', 'jpg', 'bmp', 'png']
print(os.listdir(DataDir))
print(os.listdir(os.path.join(DataDir, 'Beyonce',)))


# In[3]:


for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir,image_class)):
        print(image)


# In[4]:


total_images = 0  # initialize counter variable
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        total_images += 1  # increment counter for each file
print("Total number of images:", total_images)


# In[5]:


total_images = 0  # initialize counter variable
removed_images = 0  # initialize counter variable for removed images
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        ImagePath = os.path.join(DataDir, image_class,image)
        try:
            DImage = cv2.imread(ImagePath)
            ImaS = imghdr.what(ImagePath)
            if ImaS not in ImageEx:
                print('Image not in extension list {}'.format(ImagePath))
                os.remove(ImagePath)
                removed_images += 1  # increment counter for each removed file
        except Exception as e:
            print('Issue with image {}'.format(ImagePath))
            # os.remove(ImagePath)  # Uncomment this if you want to remove the problematic images
        total_images += 1  # increment counter for each file

print("Total number of images found:", total_images)
print("Total number of images removed:", removed_images)


# In[6]:


# Define the transforms for the dataset
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[7]:


# Load the dataset
dataset = datasets.ImageFolder('Image data', transform=transform_train)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# In[8]:


print(train_size)


# In[9]:


# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


# In[10]:


print(train_loader)


# In[11]:


# Define the device (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[12]:


import torch.nn as nn
import torchvision.models as models

class ImageClassificationModule(nn.Module):
    def __init__(self):
        super(ImageClassificationModule, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


# In[13]:


# Initialize the model and move it to the device
from torchsummary import summary
model = ImageClassificationModule().to(device)

summary(model, input_size=(3, 224, 224))


# In[14]:


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[17]:


# Train the model
for epoch in range(3):
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0

    # Training loop
    model.train()
    for inputs, labels in train_loader:\
        inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)


    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    train_correct += (predicted == labels).sum().item()


# In[18]:


# Validation loop
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_correct += (predicted == labels).sum().item()


# In[19]:


# Print statistics
train_loss /= len(train_loader.dataset)
train_acc = train_correct / len(train_loader.dataset)
val_loss /= len(val_loader.dataset)
val_acc = val_correct / len(val_loader.dataset)


# In[20]:


print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc:    {val_acc:.4f}')


# In[21]:


# Test the model on the test set
test_loss = 0.0
test_correct = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset) * 100.0
    print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.2f}%')


# In[22]:


# save the model
torch.save(model.state_dict(), 'Models/pytorchmodel.pth')

