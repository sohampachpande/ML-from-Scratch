import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Neural Network Class
class net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(net, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load Data
train = datasets.FashionMNIST('datasets/FashionMNIST',
                              train=True,
                              download=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]))

test = datasets.FashionMNIST('datasets/FashionMNIST',
                             train=False,
                             download=True,
                             transform=transforms.Compose(
                                 [transforms.ToTensor()]))

trainSet = torch.utils.data.DataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True)
testSet = torch.utils.data.DataLoader(test,
                                      batch_size=batch_size,
                                      shuffle=True)

# Initialize network
model = net(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
CEloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainSet):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0],
                            -1)  #(batchsize, 28*28 i.e. flatten image)

        # forward propagation
        scores = model(data)
        loss = CEloss(scores, targets)

        # backward propagation
        # Pytorch accumulates gradients for every backward pass. We manually reset it every epoch
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    print(loss)


# Check accuracy on training & test to see how good our model
def check_accuracy(dataloader, model):
    if dataloader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=device)
            y = y.to(device=device)
            X = X.reshape(X.shape[0], -1)

            scores = model(X)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print("Accuracy: {}".format(
            round(float(num_correct) / float(num_samples) * 100), 2))


check_accuracy(testSet, model)