import pandas as pd
from sklearn.model_selection import train_test_split

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.manual_seed(0)


def split_data(file_path, test_size=0.2, random_state=0):

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data["default payment next month"]

    # Split the data into training+validation and testing sets
    X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Calculate the proportion for the validation set
    dev_per = X_test.shape[0] / X_new.shape[0]

    # Split the training+validation set into training and validation sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=random_state)

    X_train_torch = torch.tensor(X_train.values).float().to(device)
    y_train_torch = torch.tensor(y_train.values).to(device)
    X_dev_torch = torch.tensor(X_dev.values).float().to(device)
    y_dev_torch = torch.tensor(y_dev.values).to(device)
    X_test_torch = torch.tensor(X_test.values).float().to(device)
    y_test_torch = torch.tensor(y_test.values).to(device)


    print("Training sets:", X_train.shape, y_train.shape)
    print("Validation sets:", X_dev.shape, y_dev.shape)
    print("Testing sets:", X_test.shape, y_test.shape)

    return X_train_torch, X_dev_torch, X_test_torch, y_train_torch, y_dev_torch, y_test_torch, X_train, y_train