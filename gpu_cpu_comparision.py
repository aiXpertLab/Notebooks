import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn, optim
import torch.nn.functional as F
import time

# Load dataset
data = pd.read_csv("CCDefault/a301_dccc_prepared.csv")
X = data.iloc[:, :-1]
y = data["default payment next month"]

# Artificially increase the dataset size by duplicating the data multiple times
X = pd.concat([X] * 20, ignore_index=True)
y = pd.concat([y] * 20, ignore_index=True)

X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0] / X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=0)

# Convert data to tensors
X_train_torch = torch.tensor(X_train.values).float()
y_train_torch = torch.tensor(y_train.values)
X_dev_torch = torch.tensor(X_dev.values).float()
y_dev_torch = torch.tensor(y_dev.values)

class LargerNN(nn.Module):
    def __init__(self, input_size):
        super(LargerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x

def train_model(device, X_train, y_train, X_dev, y_dev):
    model = LargerNN(X_train.shape[1]).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    batch_size = 512
    model.train()

    start_time = time.time()
    for e in range(epochs):
        X_, y_ = shuffle(X_train, y_train)
        for i in range(0, len(X_), batch_size):
            X_batch = X_[i:i+batch_size].to(device)
            y_batch = y_[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    return end_time - start_time

# Train on CPU
device_cpu = torch.device("cpu")
cpu_time = train_model(device_cpu, X_train_torch, y_train_torch, X_dev_torch, y_dev_torch)
print(f"Training time on CPU: {cpu_time:.2f} seconds")

# Train on GPU
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_time = train_model(device_gpu, X_train_torch, y_train_torch, X_dev_torch, y_dev_torch)
print(f"Training time on GPU: {gpu_time:.2f} seconds")
