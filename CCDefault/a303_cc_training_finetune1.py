# The a301 model is suffering from high bias, the focus should be on
# increasing the number of epochs or increasing the size of the network by adding
# additional layers or units to each layer. The aim should be to approximate the
# accuracy over the validation set to 80%.

import sklearn as sk
import torch

from a301_cc_tensor_dataset import split_data
from a301_cc_architecture_4_layer import Classifier
from a301_cc_matplotlib   import plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.manual_seed(0)

X_train_torch, X_dev_torch, X_test_torch, y_train_torch, y_dev_torch, y_test_torch, X_train = split_data("a301_dccc_prepared.csv")

# training
model = Classifier(X_train.shape[1]).to(device)
print(model)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
batch_size = 128
train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

def training():
    for e in range(epochs):
        model.train()
        X_, y_ = sk.utils.shuffle(X_train_torch, y_train_torch)
        running_loss = 0
        running_acc = 0
        iterations = 0

        for i in range(0, len(X_), batch_size):
            iterations += 1
            b = i + batch_size
            X_batch = X_[i:b]
            y_batch = y_[i:b]

            # pred = model(X_batch)
            log_ps = model(X_batch)
            # loss = criterion(pred, y_batch)
            loss = criterion(log_ps, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(log_ps)
            # ps = torch.exp(pred)
            top_p, top_class = ps.topk(1, dim=1)
            running_acc += sk.metrics.accuracy_score(y_batch.cpu(), top_class.cpu())

        dev_loss = 0
        acc = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            log_dev = model(X_dev_torch)
            dev_loss = criterion(log_dev, y_dev_torch)

            ps_dev = torch.exp(log_dev)
            top_p, top_class_dev = ps_dev.topk(1, dim=1)
            acc = sk.metrics.accuracy_score(y_dev_torch, top_class_dev)

        if e%50 == 0 or e == 1:
            x_axis.append(e)

            train_losses.append(running_loss/iterations)
            dev_losses.append(dev_loss)
            train_acc.append(running_acc/iterations)
            dev_acc.append(acc)

            print("Epoch: {}/{}.. ".format(e, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/iterations),
                  "Validation Loss: {:.3f}.. ".format(dev_loss),
                  "Training Accuracy: {:.3f}.. ".format(running_acc/iterations),
                  "Validation Accuracy: {:.3f}".format(acc))

if __name__ == "__main__":
    training()
    checkpoint = {"input": X_train.shape[1],"state_dict": model.state_dict()}
    torch.save(checkpoint, "./checkpoints/checkpoint_303_1.pth")
    plotting(first=train_losses, second=dev_losses)
    plotting(first=train_acc, second=dev_acc, firstlabel="Training accuracy", secondlabel="Validation accuracy")
