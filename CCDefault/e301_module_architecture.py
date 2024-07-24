# Model Architecture
# Layers: Defines the types and arrangement of layers (e.g., dense, convolutional, recurrent).
# Connections: Specifies how data flows between layers.
# Activation Functions: Applies functions to the output of layers (e.g., ReLU, Sigmoid).
# Data Flow: Describes the sequence of operations and transformations applied to the input data.
# The architecture focuses on the structure and design of the model, including how it processes inputs and produces outputs.

import torch
import torch.nn as nn
import torch.nn.functional as F

D_i = 10    # D_i refers to the input dimensions (the features in the input data),
D_h = 5     # D_h refers to the hidden dimensions (the number of nodes in a hidden layer),
D_o = 2     # D_o refers to the output dimensions.


class Classifier(torch.nn.Module):
    def __init__(self, D_i, D_h, D_o):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(D_i, D_h)
        self.linear2 = torch.nn.Linear(D_h, D_o)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        o = F.softmax(self.linear2(z))
        return o

model = Classifier(D_i, D_h, D_o)
print(model)


