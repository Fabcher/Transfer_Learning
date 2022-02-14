
from pickletools import optimize
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  numpy as np
import  torchvision
import  torchvision.models as ready_models
import  torchvision.transforms as transforms
import  matplotlib.pyplot as plt 
from torch.optim import lr_scheduler

import  time
import  os
import  copy 

#----------------------------------------------------
data_dir = './data/hymenoptera_data'
batch_size = 4
learning_rate = 0.001
num_epochs = 25


# Define data transforms
img_mean = [0.5, 0.5, 0.5]
img_std = [0.25, 0.25, 0.25]

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(img_mean,img_std)])

valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
						transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize(img_mean,img_std)])

train_dataset = torchvision.datasets.ImageFolder(data_dir+'/train',train_transforms)
valid_dataset = torchvision.datasets.ImageFolder(data_dir+'/val',valid_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

train_size = len(train_dataset)
print("Train size :",train_size)
valid_size = len(valid_dataset)
print("Valid size :",valid_size)
class_names = train_dataset.classes
num_classes = len(train_dataset.classes)
print(f'Classes : {num_classes}  {class_names}' )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print('Using ',device)

def imshow(inp, title):  # Imshow for pytorch Tensors
    inp = inp.numpy().transpose((1, 2, 0))
    inp = img_std * inp + img_mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# Get one batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs):
	print('Starting training...')
	time0 = time.time()

	best_acc = 0

	for epoch in range (num_epochs):

		model.train()  # Set model to training mode

		with torch.enable_grad():
			cnt = 0
			for images, labels in train_loader:

				images = images.to(device)
				labels = labels.to(device)

				outputs = model(images)  # Propagate forward
				loss = criterion(outputs, labels)

				optimizer.zero_grad()  # Propagate backward
				loss.backward()
				optimizer.step()

				cnt += 1
				if (cnt%10==0):
					print('.',end='')

		scheduler.step()

		model.eval()  # Set model to evaluation mode
		with torch.no_grad():

			n_correct = 0
			n_samples = 0
			n_class_correct = [0 for i in range(num_classes)]
			n_class_samples = [0 for i in range(num_classes)]

			for images, labels in valid_loader:

				images = images.to(device)
				labels = labels.to(device)

				outputs = model(images)  # Propagate forward
				_, predicted = torch.max(outputs,dim=1)  # 1 = horizontal
				loss = criterion(outputs, labels)

				n_correct += (predicted == labels).sum().item()
				n_samples += labels.size(0)

			tot_acc = 100.0 * n_correct / n_samples
			print(f'\nEpoch : {epoch},  Test accuracy : {tot_acc:.2f} %')


# Create network starting from prebuilt and pretrained model

network = ready_models.resnet18(pretrained=True)
#print(network.modules)

# We want to substitute the last layer, called fc, with our own
# We need to find out what is the size of the input
		
num_feat_into_fc = network.fc.in_features

# We can just reassign the fc layer to a newly defined one:

network.fc = nn.Linear(num_feat_into_fc, num_classes)

network.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(network.parameters(), lr = learning_rate)  # Train all parameters

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

network = train_model(network, criterion, optimizer, step_lr_scheduler, num_epochs)
