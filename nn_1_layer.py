import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

input_units = 10
output_units =1

model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid())

print(model)

loss_funct = nn.MSELoss()

print(loss_funct)

