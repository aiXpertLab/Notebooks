# custom module's methodology.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from  utils import tools
torch.manual_seed(0)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data = pd.read_csv("CCDefault/a301_dccc_prepared.csv")
X = data.iloc[:,:-1]
y = data["default payment next month"]

X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0]/X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=0)

print("Training sets:",X_train.shape, y_train.shape)
print("Validation sets:",X_dev.shape, y_dev.shape)
print("Testing sets:",X_test.shape, y_test.shape)

X_dev_torch = torch.tensor(X_dev.values).float().to(device)
y_dev_torch = torch.tensor(y_dev.values).to(device)
X_test_torch = torch.tensor(X_test.values).float().to(device)
y_test_torch = torch.tensor(y_test.values).to(device)


class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 10)
        self.hidden_2 = nn.Linear(10, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.output   = nn.Linear(10, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        out = F.log_softmax(self.output(z), dim=1)

        return out

model = Classifier(X_train.shape[1]).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
batch_size = 128

train_losses, dev_losses, train_acc, dev_acc= [], [], [], []

@tools.time_it()
def train_model():
    for e in range(epochs):
        X_, y_ = shuffle(X_train, y_train)
        running_loss = 0
        running_acc = 0
        iterations = 0

        for i in range(0, len(X_), batch_size):
            iterations += 1
            b = i + batch_size
            X_batch = torch.tensor(X_.iloc[i:b,:].values).float().to(device)
            y_batch = torch.tensor(y_.iloc[i:b].values).to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(pred)
            top_p, top_class = ps.topk(1, dim=1)
            running_acc += accuracy_score(y_batch.cpu(), top_class.cpu())

        dev_loss = 0
        acc = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            pred_dev = model(X_dev_torch)
            dev_loss = criterion(pred_dev, y_dev_torch)

            ps_dev = torch.exp(pred_dev)
            top_p, top_class_dev = ps_dev.topk(1, dim=1)
            acc = accuracy_score(y_dev_torch.cpu(), top_class_dev.cpu())

        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss.item())
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/iterations),
            "Validation Loss: {:.3f}.. ".format(dev_loss),
            "Training Accuracy: {:.3f}.. ".format(running_acc/iterations),
            "Validation Accuracy: {:.3f}".format(acc))


    fig = plt.figure(figsize=(15, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot([loss.cpu().item() if torch.is_tensor(loss) else loss for loss in dev_losses], label='Validation loss')  # Change this line
    plt.legend(frameon=False, fontsize=15)
    # plt.show()


    # fig = plt.figure(figsize=(15, 5))
    # plt.plot(train_acc, label="Training accuracy")
    # plt.plot(dev_acc, label="Validation accuracy")
    # plt.legend(frameon=False, fontsize=15)
    # # plt.show()

    # model.eval()
    # test_pred = model(X_test_torch)
    # test_pred = torch.exp(test_pred)
    # top_p, top_class_test = test_pred.topk(1, dim=1)
    # acc_test = accuracy_score(y_test_torch.cpu(), top_class_test.cpu())
    # print(acc_test)

    # checkpoint = {"input": X_train.shape[1],"state_dict": model.state_dict()}
    # torch.save(checkpoint, "./checkpoints/checkpoint.pth")

train_model()
