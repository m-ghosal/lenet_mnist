import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

activations = {}
def get_activation(name):
        def hook(model, input, output):
                activations[name] = output.detach()
        return hook

transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
train_dataset = datasets.MNIST(root = 'mnist_data', train=True, transform=transform, download=True)
valid_dataset = datasets.MNIST(root = 'mnist_data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)

plt.imshow(train_dataset.data[0], cmap='gray')
plt.title('%i' % train_dataset.targets[0])
plt.show()

fig = plt.figure()
cols, rows = 10, 5
for index in range (1, cols*rows + 1):
	sample_index = torch.randint(len(train_dataset), size=(1,)).item()
	image, label = train_dataset[sample_index]
	plt.subplot(rows, cols, index)
	plt.title(label)
	plt.axis('off')
	plt.imshow(image.squeeze(), cmap = 'gray_r')
fig.suptitle('MNIST sample train images')
plt.show()

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1)
		self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
		self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1)
		self.linear1 = nn.Linear(120, 84)
		self.linear2 = nn.Linear(84, 10)
		self.tanh = nn.Tanh()
		self.avgpool = nn.AvgPool2d(kernel_size = 2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.tanh(x)
		x = self.avgpool(x)
		x = self.conv2(x)
		x = self.tanh(x)
		x = self.avgpool(x)
		x = self.conv3(x)
		x = self.tanh(x)

		x = torch.flatten(x, 1)
		
		x = self.linear1(x)
		x = self.tanh(x)
		x = self.linear2(x)		
		
		output = F.softmax(x, dim = 1)
		return x, output

torch.manual_seed(42)
model = LeNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

def train(train_loader, model, criterion, optimizer, device):
	model.train()
	running_loss = 0
		
	for X, Y  in train_loader:
		optimizer.zero_grad()
		X = X.to(device)
		Y = Y.to(device)

		output, _ = model(X)
		loss = criterion(output, Y)
		running_loss += loss.item() * X.size(0)

		loss.backward()
		optimizer.step()
		
	epoch_loss = running_loss / len(train_loader.dataset)
	return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
	model.eval()
	running_loss = 0
		
	for X, Y in valid_loader:
		X = X.to(device)
		Y = Y.to(device)

		output, _ = model(X)
		loss = criterion(output, Y)
		running_loss += loss.item() * X.size(0)
		
	epoch_loss = running_loss / len(valid_loader.dataset)
	return model, epoch_loss

def check_accuracy(model, data_loader, device):
	correct_pred = 0
	total_samples = 0

	with torch.no_grad():
		model.eval()
		for X, Y in data_loader:
			X = X.to(device)
			Y = Y.to(device)

			_, output = model(X)
			_, predict = torch.max(output, 1)
			
			correct_pred += (predict == Y).sum()
			total_samples += Y.size(0)
	
	return (correct_pred / total_samples) * 100 

def plot_losses(train_losses, valid_losses):
	train_losses = np.array(train_losses)
	valid_losses = np.array(valid_losses)

	fig, ax = plt.subplots(figsize = (12, 6.5))

	ax.plot(train_losses, color = 'black', label = 'Training loss')
	ax.plot(valid_losses, color = 'red', label = 'Validation loss')
	ax.set(title = "Training/Validation Loss", xlabel='Epoch', ylabel='Loss')
	ax.legend()
	fig.show()
	plt.show()

def plot_accuracies(train_accuracies, valid_accuracies):		
	train_accuracies = np.array(train_accuracies)
	valid_accuracies = np.array(valid_accuracies)

	fig, ax = plt.subplots(figsize = (12, 6.5))

	ax.plot(train_accuracies, color = 'black', label = 'Training Accuracy')
	ax.plot(valid_accuracies, color = 'red', label = 'Validation Accuracy')
	ax.set(title = "Training/Validation Accuracy", xlabel='Epoch', ylabel='Accuracy')
	ax.legend()
	fig.show()
	plt.show()

def loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every_epoch = 1):
	best_loss = 1e10
	train_losses = []
	valid_losses = []
	train_accuracies = []
	valid_accuracies = []		

	for epoch in range(0, epochs):
		model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
		train_losses.append(train_loss)
			
		with torch.no_grad():
			model, valid_loss = validate(valid_loader, model, criterion, device)
			valid_losses.append(valid_loss)
		if epoch % print_every_epoch == (print_every_epoch - 1):
			train_accuracy = check_accuracy(model, train_loader, device = device)			
			train_accuracies.append(train_accuracy) 
			valid_accuracy = check_accuracy(model, valid_loader, device = device)
			valid_accuracies.append(valid_accuracy)
			print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {train_accuracy:.2f}\t'
                  f'Valid accuracy: {valid_accuracy:.2f}')

	plot_losses(train_losses, valid_losses)
	plot_accuracies(train_accuracies, valid_accuracies)
	return model, optimizer, (train_losses, valid_losses), (train_accuracies, valid_accuracies)

model.linear1.register_forward_hook(get_activation('linear1'))
model.linear2.register_forward_hook(get_activation('linear2'))

model, optimizer, _, _ = loop(model, criterion, optimizer, train_loader, valid_loader, 15, device)

summary(model, (1, 32, 32))

model.linear1.register_forward_hook(get_activation('linear1'))
model.linear2.register_forward_hook(get_activation('linear2'))

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))

data, _ = valid_dataset[0]
data.unsqueeze_(0)
output, _ = model(data)

print(activations['linear2'])
plt.matshow(activations['linear1'])
plt.show()
plt.matshow(activations['linear2'])
plt.show()

activation_1 = activations['conv1'].squeeze()
activation_2 = activations['conv2'].squeeze()
activation_3 = activations['conv3'].squeeze()
activation_3 = torch.unsqueeze(activation_3, dim=-1)
activation_3 = torch.unsqueeze(activation_3, dim=-1)

def featuremap(name):
	fig, ax = plt.subplots(name.size(0))
	for i in range(name.size(0)):			
		ax[i].imshow(name[i])
		ax[i].set_axis_off()
	plt.show()

featuremap(activation_3)
featuremap(activation_2)
featuremap(activation_1)

y_predicted = []
y_true = []
for X, Y in valid_loader:
	_, output = model(X)
	output = (torch.max(torch.exp(output), 1)[1]).data.numpy()
	y_predicted.extend(output)
	labels = Y.data.numpy()
	y_true.extend(labels)

cf_matrix = confusion_matrix(y_true, y_predicted)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
df_cf_matrix = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (12, 7))
sn.heatmap(df_cf_matrix, annot = True)
plt.show()

fig = plt.figure()
for index in range(1, 51):
	plt.subplot(5, 10, index)
	plt.axis('off')
	plt.imshow(valid_dataset.data[index], cmap='gray_r')

	with torch.no_grad():
		model.eval()
		_, probs = model(valid_dataset[index][0].unsqueeze(0))
		
	title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f})%)'
	plt.title(title, fontsize=7)

fig.suptitle('Sample Predictions')
fig.show()
plt.show()
