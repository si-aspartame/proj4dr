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

        self.fc1 = nn.Linear(self.unit, self.unit*6)
        self.bn1 = nn.BatchNorm1d(self.unit*6)

        self.fc2 = nn.Linear(self.unit*6, self.unit*5)
        self.bn2 = nn.BatchNorm1d(self.unit*5)

        self.fc3 = nn.Linear(self.unit*5, self.unit*4)
        self.bn3 = nn.BatchNorm1d(self.unit*4)

        self.fc4 = nn.Linear(self.unit*4, self.unit*4)
        self.bn4 = nn.BatchNorm1d(self.unit*4)

        self.fc5 = nn.Linear(self.unit*4, self.unit*3)
        self.bn5 = nn.BatchNorm1d(self.unit*3)

        self.fc6 = nn.Linear(self.unit*3, self.unit*3)
        self.bn6 = nn.BatchNorm1d(self.unit*3)

        self.fc7 = nn.Linear(self.unit*3, self.unit*2)
        self.bn7 = nn.BatchNorm1d(self.unit*2)

        self.fc8 = nn.Linear(self.unit*2, self.unit*2)
        self.bn8 = nn.BatchNorm1d(self.unit*2)

        self.fc9 = nn.Linear(self.unit*2, self.unit)
        self.bn9 = nn.BatchNorm1d(self.unit)

        self.fc10 = nn.Linear(self.unit, 2)

        self.silu = nn.SiLU()

    def forward(self, x):
        x = x.view(-1, self.unit)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.silu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.silu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.silu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.silu(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.silu(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = self.silu(x)

        x = self.fc7(x)
        x = self.bn7(x)
        x = self.silu(x)

        x = self.fc8(x)
        x = self.bn8(x)
        x = self.silu(x)

        x = self.fc9(x)
        x = self.bn9(x)
        x = self.silu(x)

        x = self.fc10(x)

        return x

def remove_diagonal(tensor):
    rows, cols = tensor.size()
    without_diag = tensor[~torch.eye(rows, dtype=bool)].reshape(rows, cols - 1)
    return without_diag

def calculate_P(X, precomputed_beta, cpd_distances, restore_tensor):
    with torch.inference_mode():
        D = C.get_distance_matrix_of_batch_optimized(X, cpd_distances, restore_tensor)
        D = remove_diagonal(D)
        # 事前計算された beta を使用
        beta = precomputed_beta.view(-1, 1)
        P = torch.exp(-D * beta)
        P = P / torch.sum(P)
        return P + 1e-20

def calculate_Q(Y):
    D = 1 + torch.cdist(Y, Y, p=2)**2
    D = remove_diagonal(D)
    Q = (1 / (1 + D))
    Q = Q / torch.sum(Q)
    return Q