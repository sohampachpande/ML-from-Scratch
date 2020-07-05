import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# ConvolutionalNeuralNetwork
class cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten for fully connected layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load Data
train = datasets.FashionMNIST('datasets/FashionMNIST',
                              train=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]))

test = datasets.FashionMNIST('datasets/FashionMNIST',
                             train=False,
                             transform=transforms.Compose(
                                 [transforms.ToTensor()]))

trainSet = torch.utils.data.DataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True)
testSet = torch.utils.data.DataLoader(test,
                                      batch_size=batch_size,
                                      shuffle=True)

# Initialize network
model = cnn(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
CEloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainSet):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

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

            scores = model(X)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print("Accuracy: {}".format(
            round(float(num_correct) / float(num_samples) * 100), 2))


check_accuracy(testSet, model)