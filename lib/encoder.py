import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.calc as C

#%%s
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_columns):
        super(Net, self).__init__()
        self.unit = n_columns

        # 各層の設定
        self.fc1 = nn.Linear(self.unit, self.unit * 400)
        self.bn1 = nn.BatchNorm1d(self.unit * 400)

        self.fc2 = nn.Linear(self.unit * 400, self.unit * 200)
        self.bn2 = nn.BatchNorm1d(self.unit * 200)

        self.fc3 = nn.Linear(self.unit * 200, self.unit * 100)
        self.bn3 = nn.BatchNorm1d(self.unit * 100)

        self.fc4 = nn.Linear(self.unit * 100, self.unit * 50)
        self.bn4 = nn.BatchNorm1d(self.unit * 50)

        self.fc5 = nn.Linear(self.unit * 50, self.unit * 25)
        self.bn5 = nn.BatchNorm1d(self.unit * 25)

        self.fc6 = nn.Linear(self.unit * 25, self.unit * 12)
        self.bn6 = nn.BatchNorm1d(self.unit * 12)

        self.fc7 = nn.Linear(self.unit * 12, self.unit * 6)
        self.bn7 = nn.BatchNorm1d(self.unit * 6)

        self.fc8 = nn.Linear(self.unit * 6, self.unit * 4)
        self.bn8 = nn.BatchNorm1d(self.unit * 4)

        self.fc9 = nn.Linear(self.unit * 4, self.unit * 3)
        self.bn9 = nn.BatchNorm1d(self.unit * 3)

        self.fc10 = nn.Linear(self.unit * 3, self.unit * 2)
        self.bn10 = nn.BatchNorm1d(self.unit * 2)

        self.fc11 = nn.Linear(self.unit * 2, 20)

    def forward(self, x):
        x = x.view(-1, self.unit)

        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.bn5(self.fc5(x)))
        x = torch.relu(self.bn6(self.fc6(x)))
        x = torch.relu(self.bn7(self.fc7(x)))
        x = torch.relu(self.bn8(self.fc8(x)))
        x = torch.relu(self.bn9(self.fc9(x)))
        x = torch.relu(self.bn10(self.fc10(x)))
        x = self.fc11(x)

        return x

def remove_diagonal(tensor):
    rows, cols = tensor.size()
    without_diag = tensor[~torch.eye(rows, dtype=bool)].reshape(rows, cols - 1)
    return without_diag

def calculate_P(X, cpd_distances, restore_tensor):
    D = C.get_distance_matrix_of_batch_optimized(X, cpd_distances, restore_tensor)
    return D

def calculate_Q(Y):
    D = torch.cdist(Y, Y, p=1)
    return D