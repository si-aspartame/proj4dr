#%%
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.calc as C
import lib.encoder as E
import lib.util as U
import torch.nn.functional as F

import pickle

#%%
# Initialize CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
data_name = "adult"
batch_size = 64
nb_epoch = 1000
perplexity = 50
learning_rate = 1e-3
max_patience = 10  # Maximum number of epochs to wait for improvement
percentage_as_neighbor = 0.05

#%%
bin_size = 20
low_freq_threshold = 0

#%%
original_df = U.loading_data(data_name)#datasets.MNIST('./data', train=True, download=True, transform=transform)
original_df = original_df.iloc[:len(original_df)-(len(original_df)%10)]#データの長さは10で割り切れるようにする
start_time = time.time()
df = U.equal_frequency_binning(original_df.copy(), bin_size) # 数値データのカテゴリ化
df = U.replace_low_frequency_values(df, threshold=low_freq_threshold)
df = U.encode_object_column(df)
df = U.label_by_counts(df)
df, restore_tensor = U.normalize_all_columns(df)
restore_tensor.to(device)
df = df.loc[:, (df != df.iloc[0]).any()]#無駄な列を落とす
df = U.reduce_mem_usage(df)
df.describe()
label_column = df.columns[-1]
df.to_csv("original.csv")
# %%
list_of_entire_fdt = {n: df[c].value_counts() for n, c in enumerate(df.columns)}
all_calculated_cpd = C.calculate_all_cpd(df, list_of_entire_fdt, restore_tensor)

#%%
cpd_distances = C.calculate_cpd_distances(df, all_calculated_cpd, restore_tensor).to(device)
print(f"{U.get_tensor_memory_size(cpd_distances)} GB")

#%%
train_dataset = U.dataframe_to_dataset(df, label_column=label_column)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = E.Net(len(df.columns)-1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#%%
print("Training Net..")
print(f"total_iteration:{int(len(df)/batch_size)}it")
best_train_loss = float('inf')  #Initialize the best validation loss to infinity
patience = 0  #Initialize patience counter
mse = nn.MSELoss()
for epoch in range(nb_epoch):
    total_loss = 0  # Initialize total loss for this epoch
    num_batches = 0  # Initialize batch count for this epoch
    for i, (data, target) in tqdm(enumerate(train_loader)):
        data = data.view(data.shape[0], -1).to(device)
        # Pの計算にbatch_betaを使用
        P = E.calculate_P(data, cpd_distances, restore_tensor).to(device)  # 高次元空間での確率分布
        optimizer.zero_grad()
        output = model(data)
        Q = E.calculate_Q(output).to(device)#低次元空間での確率分布
        # kl_per_point = torch.sum(P * (torch.log(P) - torch.log(Q)), dim=1)
        # loss = torch.mean(kl_per_point)
        loss = mse(P, Q)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()  # Accumulate the loss
        num_batches += 1  # Increment the batch count

    avg_loss = total_loss / num_batches  # Calculate the average loss for this epoch
    print(f"Epoch {epoch+1}/{nb_epoch}, Average Loss: {avg_loss}")
    print(P[1, :3])
    print(Q[1, :3])

    # Early Stopping Check based on Training Loss
    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        patience = 0  # Reset patience counter
    else:
        patience += 1  # Increment patience counter
        if patience >= max_patience:
            print("Early stopping due to lack of improvement in training loss.")
            break

#%%
all_outputs = []
all_targets = []

with torch.no_grad():
    for i, (data, target) in enumerate(DataLoader(train_dataset, batch_size=batch_size, shuffle=False)):
        data = data.view(-1, len(df.columns)-1).to(device)  # Add .to(device)
        target = target.to(device)  # Add this if target is also used on the device
        output = model(data)
        all_outputs.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# all_outputs は既に準備されていると仮定
# original_df の各カラムを使って all_outputs の点を着色
all_outputs = np.concatenate(all_outputs, axis=0)
all_outputs = np.array(all_outputs)
print(all_outputs.shape)
np.savetxt("array.csv", all_outputs, delimiter=",")