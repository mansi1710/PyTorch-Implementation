import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
import torchvision.datasets as dsets 
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import time 

## transformers

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

data_dir = r'C:\Users\Lenovo\Anaconda3\hymenoptera_data'

train_dataset= dsets.ImageFolder(os.path.join(data_dir, 'train'), transform= data_transforms['train'])
test_dataset= dsets.ImageFolder(os.path.join(data_dir, 'val'), transform= data_transforms['val']);

print(type(train_dataset))
image, label= train_dataset[1];
plt.imshow(image[0])
plt.show()

train_data_loader= torch.utils.data.DataLoader(train_dataset, batch_size=4,
	shuffle=True, num_workers=4)

test_data_loader= torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=4)
class_name= train_dataset.classes;
print(class_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train a classifier

model= models.resnet18(pretrained= True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')