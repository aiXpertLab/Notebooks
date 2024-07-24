import sklearn as sk
import torch

from a301_cc_tensor_dataset import split_data
from a301_cc_architecture_4_layer import Classifier
from a301_cc_matplotlib   import plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

X_train_torch, X_dev_torch, X_test_torch, y_train_torch, y_dev_torch, y_test_torch, X_train = split_data("a301_dccc_prepared.csv")

# training
model = Classifier(X_train.shape[1]).to(device)
print(model)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 128
train_losses, dev_losses, train_acc, dev_acc = [], [], [], []


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

            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(pred)
            top_p, top_class = ps.topk(1, dim=1)
            running_acc += sk.metrics.accuracy_score(y_batch.cpu(), top_class.cpu())

        model.eval()
        with torch.no_grad():
            pred_dev = model(X_dev_torch)
            dev_loss = criterion(pred_dev, y_dev_torch)

            ps_dev = torch.exp(pred_dev)
            top_p, top_class_dev = ps_dev.topk(1, dim=1)
            acc = sk.metrics.accuracy_score(y_dev_torch.cpu(), top_class_dev.cpu())

        train_losses.append(running_loss / iterations)
        dev_losses.append(dev_loss.item())
        train_acc.append(running_acc / iterations)
        dev_acc.append(acc)

        print(f"Epoch: {e+1}/{epochs}.. ",
              f"Training Loss: {running_loss / iterations:.3f}.. ",
              f"Validation Loss: {dev_loss.item():.3f}.. ",
              f"Training Accuracy: {running_acc / iterations:.3f}.. ",
              f"Validation Accuracy: {acc:.3f}")

if __name__ == "__main__":
    training()
    checkpoint = {"input": X_train.shape[1],"state_dict": model.state_dict()}
    torch.save(checkpoint, "./checkpoints/checkpoint.pth")
    plotting(first=train_losses, second=dev_losses)
    plotting(first=train_acc, second=dev_acc, firstlabel="Training accuracy", secondlabel="Validation accuracy")
