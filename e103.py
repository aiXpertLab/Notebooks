import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

input_units = 10
output_units =1

model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid())
loss_funct = nn.MSELoss()

x = torch.randn(20,10)
y = torch.randint(0,2, (20,1)).type(torch.FloatTensor)

optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
for i in range(20):
    y_pred = model(x)
    loss = loss_funct(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i%5 == 0:
        print(i, loss.item())
        
plt.plot(range(0,20),losses)
plt.show()