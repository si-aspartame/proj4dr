
#%%
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.calc as C
import lib.tsne as TSNE
import lib.util as U
import torch.nn.functional as F

import pickle

#%%
# Initialize CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
data_name = "creditcard"
batch_size = 1000
nb_epoch = 1000
perplexity = 50
learning_rate = 1e-3
max_patience = 10  # Maximum number of epochs to wait for improvement
percentage_as_neighbor = 0.05

#%%
bin_size = 30
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

# %%
list_of_entire_fdt = {n: df[c].value_counts() for n, c in enumerate(df.columns)}
all_calculated_cpd = C.calculate_all_cpd(df, list_of_entire_fdt, restore_tensor)

#%%
cpd_distances = C.calculate_cpd_distances(df, all_calculated_cpd, restore_tensor).to(device)
print(f"{U.get_tensor_memory_size(cpd_distances)} GB")

#%%
if os.path.exists(f'precomputed_beta_{batch_size}_{perplexity}_{int(percentage_as_neighbor*100)}.pkl'):
    # 保存されたprecomputed_betaを読み込む
    with open(f'precomputed_beta_{batch_size}_{perplexity}_{int(percentage_as_neighbor*100)}.pkl', 'rb') as f:
        precomputed_beta_values = pickle.load(f)
    df['precomputed_beta'] = precomputed_beta_values
else:
    # precomputed_betaが存在しない場合、新しく計算
    #int(len(df)*percentage_as_neighbor)
    df['precomputed_beta'] = C.compute_beta2(df, perplexity, int(len(df)*percentage_as_neighbor), cpd_distances, restore_tensor)
    # そして保存
    with open(f'precomputed_beta_{batch_size}_{perplexity}_{int(percentage_as_neighbor*100)}.pkl', 'wb') as f:
        pickle.dump(df['precomputed_beta'].values, f)
print(df['precomputed_beta'].values)
# max_value = df['precomputed_beta'].max()
# second_largest_value = df['precomputed_beta'].nlargest(2).iloc[-1]
# df['precomputed_beta'] = df['precomputed_beta'].replace(max_value, second_largest_value)

#%%
df.sort_values(by='precomputed_beta', ascending=False).head(100)


#%%
train_dataset = U.dataframe_to_dataset(df, label_column=label_column)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TSNE.Net(len(df.columns)-1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#%%
print("Training Net..")
print(f"total_iteration:{int(len(df)/batch_size)}it")
best_train_loss = float('inf')  #Initialize the best validation loss to infinity
patience = 0  #Initialize patience counter

for epoch in range(nb_epoch):
    total_loss = 0  # Initialize total loss for this epoch
    num_batches = 0  # Initialize batch count for this epoch
    for i, (data, target) in tqdm(enumerate(train_loader)):
        data = data.view(data.shape[0], -1).to(device)
        # 分離: データとbeta
        actual_data, batch_beta = data[:, :-1], data[:, -1]
        # Pの計算にbatch_betaを使用
        P = TSNE.calculate_P(actual_data, batch_beta, cpd_distances, restore_tensor).to(device)  # 高次元空間での確率分布
        optimizer.zero_grad()
        output = model(data)
        Q = TSNE.calculate_Q(output).to(device)#低次元空間での確率分布
        # kl_per_point = torch.sum(P * (torch.log(P) - torch.log(Q)), dim=1)
        # loss = torch.mean(kl_per_point)
        loss = C.js_divergence(P, Q)#C.spearman_loss(P*100000, Q*100000)#F.kl_div(torch.log(Q), P, reduction='mean')##
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()  # Accumulate the loss
        num_batches += 1  # Increment the batch count

    avg_loss = total_loss / num_batches  # Calculate the average loss for this epoch
    print(f"Epoch {epoch+1}/{nb_epoch}, Average Loss: {avg_loss}")

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

reversed_columns = original_df.columns[::-1]  # ここで列を逆順にします

if len(reversed_columns) < 30:
    n_cols = 6
    n_rows = int(np.ceil(len(reversed_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
else:
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = np.array([[axes]])

for i, column in enumerate(reversed_columns):  # ここで逆順にした列を使います
    if len(reversed_columns) >= 30:
        ax = axes[0, 0]
    else:
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

    # カラーマップの選択
    if original_df[column].dtype == 'object':  # カテゴリカルデータの場合
        cmap = plt.cm.tab20
        unique_categories = original_df[column].unique()[:10]
        
        for idx, category in enumerate(unique_categories):
            subset = all_outputs[original_df[column] == category]
            ax.scatter(subset[:, 0], subset[:, 1], color=cmap(idx / len(unique_categories)), label=str(category), s=5)

        ax.legend()
    else:  # 数値データの場合
        cmap = plt.cm.viridis
        colors = original_df[column]
        sc = ax.scatter(all_outputs[:, 0], all_outputs[:, 1], c=colors, cmap=cmap, s=5)
        fig.colorbar(sc, ax=ax)

    ax.set_title(f'{column}')

    if len(original_df.columns) >= 30:
        break  # 1つの図だけを描画したらループを抜ける

# 余分なサブプロットを削除
if len(original_df.columns) < 30 and len(original_df.columns) % n_cols != 0:
    for i in range(len(original_df.columns), n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.savefig("output.png")
plt.show()

# %%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
# %%
