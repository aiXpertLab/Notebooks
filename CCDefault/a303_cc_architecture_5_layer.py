# Model Architecture
# Layers: Defines the types and arrangement of layers (e.g., dense, convolutional, recurrent).
# Connections: Specifies how data flows between layers.
# Activation Functions: Applies functions to the output of layers (e.g., ReLU, Sigmoid).
# Data Flow: Describes the sequence of operations and transformations applied to the input data.
# The architecture focuses on the structure and design of the model, including how it processes inputs and produces outputs.

import torch


class Classifier(torch.nn.Module):
    def __init__(self, input_size, o1=10, o2=10, o3=10, o4=10):
        super().__init__()
        self.hidden_1 = torch.nn.Linear(input_size, o1)
        self.hidden_2 = torch.nn.Linear(o1, o2)
        self.hidden_3 = torch.nn.Linear(o2, o3)
        self.hidden_4 = torch.nn.Linear(o3, o4)
        self.output   = torch.nn.Linear(o4, 2)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        z = torch.nn.functional.relu(self.hidden_1(x))
        z = torch.nn.functional.relu(self.hidden_2(z))
        z = torch.nn.functional.relu(self.hidden_3(z))
        z = torch.nn.functional.relu(self.hidden_4(z))
        o = torch.nn.functional.log_softmax(self.output(z), dim=1)
        return o

