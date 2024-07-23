import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(0)
data = pd.read_csv('./data/SomervilleHappinessSurvey2015.csv')
print(data.head())

x = torch.tensor(data.iloc[:,1:].values).float()
y = torch.tensor(data.iloc[:,:1].values).float()

model = nn.Sequential(nn.Linear(6,1), nn.Sigmoid())
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

losses=[]
for i in range(100):
    y_pred = model(x)
    loss = loss_function(y_pred,y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%10 == 0:
        print(loss.item())

plt.plot(range(0,100),losses)
plt.show()