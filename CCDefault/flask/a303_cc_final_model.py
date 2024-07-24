import torch


class Classifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = torch.nn.Linear(input_size, 100)
        self.hidden_2 = torch.nn.Linear(100, 100)
        self.hidden_3 = torch.nn.Linear(100, 50)
        self.hidden_4 = torch.nn.Linear(50,50)
        self.output = torch.nn.Linear(50, 2)
        
        self.dropout = torch.nn.Dropout(p=0.1)
        #self.dropout_2 = nn.Dropout(p=0.1)
        
    def forward(self, x):
        z = self.dropout(torch.nn.functional.relu(self.hidden_1(x)))
        z = self.dropout(torch.nn.functional.relu(self.hidden_2(z)))
        z = self.dropout(torch.nn.functional.relu(self.hidden_3(z)))
        z = self.dropout(torch.nn.functional.relu(self.hidden_4(z)))
        out = torch.nn.functional.log_softmax(self.output(z), dim=1)
        
        return out
