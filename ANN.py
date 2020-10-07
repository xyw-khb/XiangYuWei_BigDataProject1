import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
test_data = open(r'new_data.csv', 'r')
reader_test = pd.read_csv(test_data)
test_data.close()
X = reader_test.iloc[:, 5:]
X = torch.from_numpy(np.array(X))

y = reader_test['winner']
y = torch.from_numpy(np.array(y - 1))

train_x = torch.as_tensor(X.clone().detach(), dtype=torch.float)
train_y = torch.as_tensor(y.clone().detach(), dtype=torch.long)

test_data = open(r'test_set.csv', 'r')
reader_test = pd.read_csv(test_data)
test_data.close()

test_x = reader_test.iloc[:, 5:]
test_y = reader_test['winner']
test_x = torch.from_numpy(np.array(test_x))
test_y = torch.from_numpy(np.array(test_y - 1))

test_x = torch.as_tensor(test_x.clone().detach(), dtype=torch.float)
test_y = torch.as_tensor(test_y.clone().detach(), dtype=torch.long)

train_set = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_set, batch_size=50, shuffle=True)

class Net(torch.nn.Module):
    #模型
    def __init__(self):
        super(Net, self).__init__()
        self.L = nn.Linear(16, 44)
        self.L2 = nn.Linear(44, 25)
        self.L3 = nn.Linear(25, 10)
        self.L4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.L(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        out = F.softmax(self.L4(x), dim=-1)
        return out

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_f = nn.CrossEntropyLoss()

for epoch in range(100):
    net.train()
    train_loss = 0.0
    for data in train_loader:
        x, y = data
        optimizer.zero_grad()
        y_p = net(x)
        loss = loss_f(y_p, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('epoch' + str(epoch) + 'loss' + str(train_loss))

with torch.no_grad():
    #预测
    y_p = net(test_x)
    value, indices = torch.topk(y_p, 1)
    le_y = len(test_y)
    count = 0

    for j in range(0, le_y):
        if float(value[j]) == float(test_y[j]):
            count = count + 1
        else:
            pass
    print("ANN模型评价:",count / le_y)
