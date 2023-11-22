from itertools import product

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import lib.util as U
import lib.encoder as E
import torch.nn.functional as F
import torchsort

def insert_zeros_into_fdt(conditional_fdt, entire_fdt):
    uniques = list(entire_fdt.keys())
    new_conditional_fdt = np.array([conditional_fdt.get(k, 0) for k in uniques])
    return uniques, new_conditional_fdt

def get_cpd(conditional_fdt, entire_fdt):
    _, fdt_count = insert_zeros_into_fdt(conditional_fdt, entire_fdt)
    fdt_count = (fdt_count / np.sum(fdt_count)) + 1e-8
    if np.sum(fdt_count) == 0:
        print("sum of cpd is zero")
        raise ValueError
    return fdt_count / np.sum(fdt_count)

def denormalize_values(restore_tensor, column_indices, normalized_values):
    device = normalized_values.device  # デバイスを取得
    restore_tensor = restore_tensor.to(device)  # restore_tensorを同じデバイスに移動
    column_indices = column_indices.to(device).to(torch.int)  # column_indicesも同じデバイスに移動
    
    # Extract min and max values
    min_vals = restore_tensor[column_indices, 0]
    max_vals = restore_tensor[column_indices, 1]
    
    # Calculate the difference between max and min values
    diffs = max_vals - min_vals
    
    # Perform the denormalization
    original_values = (min_vals + (normalized_values + 1) * diffs / 2).int()
    
    return original_values

def sort_by_counts(unique_values, counts):
    sorted_indices = np.argsort(-counts)  # countsの多い順（降順）にソートするためのインデックスを取得
    sorted_unique_values = unique_values[sorted_indices]  # unique_valuesをソート
    sorted_counts = counts[sorted_indices]  # countsをソート
    return sorted_unique_values, sorted_counts

def calculate_partial_cpd(num_c_A, num_c_B, df_np, list_of_entire_fdt, restore_tensor):
    unique_values, counts = np.unique(df_np[:, num_c_A], return_counts=True)
    num_c_A_vc = dict(zip(unique_values, counts))

    partial_cpd = {}
    
    for unique_A in num_c_A_vc.keys():
        mask = (df_np[:, num_c_A] == unique_A)
        conditional_B = df_np[mask, num_c_B]
        unique_values, counts = np.unique(conditional_B, return_counts=True)
        conditional_fdt = dict(zip(unique_values, counts))
        cpd = get_cpd(conditional_fdt, list_of_entire_fdt[num_c_B])
        partial_cpd[denormalize_values(restore_tensor, torch.Tensor([num_c_A]), torch.Tensor([unique_A])).item()] = cpd
        if np.all(cpd == cpd[0]):
            print(conditional_fdt)
    
    return num_c_A, num_c_B, partial_cpd

def calculate_all_cpd(df, list_of_entire_fdt, restore_tensor):
    print("Calculating all CPD..")
    n_columns = len(df.columns)
    df_np = df.to_numpy()
    
    all_calculated_cpd = {}
    print(f"total_iteration:{len(df.columns) ** 2}it")
    results = Parallel(n_jobs=4)(delayed(calculate_partial_cpd)(num_c_A, num_c_B, df_np, list_of_entire_fdt, restore_tensor) for num_c_A, num_c_B in tqdm(product(range(n_columns), range(n_columns))))
    
    for num_c_A, num_c_B, partial_cpd in results:
        all_calculated_cpd[(num_c_A, num_c_B)] = partial_cpd#num_c_Aの値がnのときのnum_c_BのCPD一覧
    
    return all_calculated_cpd

def calculate_cpd_distances(df, all_calculated_cpd, restore_tensor):
    print("Calculating all distances between CPDs..")
    cpd_distances = torch.zeros((len(df.columns), len(df.columns), df.nunique().max(), df.nunique().max()))
    
    for num_c_A, num_c_B in tqdm(product(list(range(len(df.columns))), list(range(len(df.columns))))):
        num_c_A_vc = df.iloc[:, num_c_A].value_counts()
        for unique_A1, unique_A2 in [(a, b) for a, b in list(product(num_c_A_vc.index.values, num_c_A_vc.index.values))]:
            unique_A1 = denormalize_values(restore_tensor, torch.Tensor([num_c_A]), torch.Tensor([unique_A1])).item()
            unique_A2 = denormalize_values(restore_tensor, torch.Tensor([num_c_A]), torch.Tensor([unique_A2])).item()

            cpd1_dict = all_calculated_cpd[(num_c_A, num_c_B)]
            cpd1 = cpd1_dict[unique_A1]

            cpd2_dict = all_calculated_cpd[(num_c_A, num_c_B)]
            cpd2 = cpd2_dict[unique_A2]

            if len(cpd1) == 0 or len(cpd2) == 0:
                continue

            if np.array_equal((cpd1*1e+6).astype(int), (cpd2*1e+6).astype(int)):
                div = 0.0#0.0#0.0#1e-4#0.0
            else:
                div = jensenshannon(cpd1, cpd2)#

            if not div >= 0.0:
                print(cpd1)
                print(cpd2)
                div = 0.0
                raise ValueError

            cpd_distances[num_c_A, num_c_B, unique_A1, unique_A2] = div
            
    return cpd_distances

def get_distance_matrix_of_batch_optimized(X, cpd_distances, restore_tensor):
    with torch.inference_mode():
        device = X.device
        n_rows, n_cols = X.shape

        denorm_X = denormalize_values(restore_tensor, torch.arange(n_cols).repeat(n_rows, 1).to(device), X).int()
        dm_upper_idx_row, dm_upper_idx_col = torch.triu_indices(n_rows, n_rows, offset=1, device=device)
        rows_n = denorm_X.index_select(0, dm_upper_idx_row)
        rows_m = denorm_X.index_select(0, dm_upper_idx_col)

        num_c_A, num_c_B = torch.meshgrid(torch.arange(n_cols, device=device), torch.arange(n_cols, device=device))
        distance_n_m = torch.sum(cpd_distances[num_c_A, num_c_B, rows_n[:, num_c_A], rows_m[:, num_c_A]], dim=(1,2))

        distance_matrix = torch.zeros(n_rows, n_rows, device=device)
        
        distance_matrix[dm_upper_idx_row, dm_upper_idx_col] = distance_n_m
        distance_matrix[dm_upper_idx_col, dm_upper_idx_row] = distance_n_m
        
        return distance_matrix

def get_distance_matrix_of_batch_optimized2(X, Y, cpd_distances, restore_tensor):
    with torch.inference_mode():
        device = X.device
        n_rows_X, n_cols_X = X.shape#48840x15
        n_rows_Y, n_cols_Y = Y.shape#4440x15

        assert n_cols_X == n_cols_Y, "Number of columns in X and Y should be the same"

        denorm_X = denormalize_values(restore_tensor, torch.arange(n_cols_X).repeat(n_rows_X, 1).to(device), X).int()
        denorm_Y = denormalize_values(restore_tensor, torch.arange(n_cols_Y).repeat(n_rows_Y, 1).to(device), Y).int()

        distance_matrix = torch.zeros(n_rows_X, n_rows_Y, device=device)

        num_c_A, num_c_B = torch.meshgrid(torch.arange(n_cols_X, device=device), torch.arange(n_cols_X, device=device))
        dm_upper_idx_row, dm_upper_idx_col = torch.triu_indices(n_rows_Y, n_rows_Y, offset=1, device=device)

        distance_matrix = torch.zeros(n_rows_X, n_rows_Y, device=device)

        for start_idx in range(0, n_rows_X, n_rows_Y):
            end_idx = min(start_idx + n_rows_Y, n_rows_X)
            batch_X = denorm_X[start_idx:end_idx]
            rows_n = batch_X.index_select(0, dm_upper_idx_row)#batch_X.shape==denorm_Y.shape
            rows_m = denorm_Y.index_select(0, dm_upper_idx_col)
            distance_n_m = torch.sum(cpd_distances[num_c_A, num_c_B, rows_n[:, num_c_A], rows_m[:, num_c_A]], dim=(1, 2))
            batch_distance_matrix = torch.zeros(n_rows_Y, n_rows_Y, device=device)
            batch_distance_matrix[dm_upper_idx_row, dm_upper_idx_col] = distance_n_m
            batch_distance_matrix[dm_upper_idx_col, dm_upper_idx_row] = distance_n_m
            distance_matrix[start_idx:end_idx, :] = batch_distance_matrix
        return distance_matrix.transpose(0, 1)

def binary_search_beta(perplexity, k_neighbors_distances, tol=1e-5, max_iter=50):
    n = k_neighbors_distances.shape[0]  # バッチサイズ
    beta = np.full(n, 2.0)  # 各データポイントに対して初期betaを設定
    beta_min = np.full(n, -np.inf)
    beta_max = np.full(n, np.inf)

    for iter in range(max_iter):
        P = np.exp(-k_neighbors_distances * beta[:, None])
        sum_P = np.sum(P, axis=1)

        mask_zero = sum_P == 0
        sum_P[mask_zero] = 1e-8
        P /= sum_P[:, None]

        H = -np.sum(P * np.log(P + 1e-8), axis=1)
        Hdiff = H - np.log(perplexity)

        mask_positive = Hdiff > 0
        mask_negative = ~mask_positive

        beta_min[mask_positive] = beta[mask_positive]
        beta_max[mask_negative] = beta[mask_negative]

        beta[mask_positive] = np.where(beta_max[mask_positive] == np.inf,
                                       beta[mask_positive] * 2.0,
                                       (beta[mask_positive] + beta_max[mask_positive]) / 2.0)

        beta[mask_negative] = np.where(beta_min[mask_negative] == -np.inf,
                                       beta[mask_negative] / 2.0,
                                       (beta[mask_negative] + beta_min[mask_negative]) / 2.0)

        if np.all(np.abs(Hdiff) <= tol):
            break

    return beta

import torch

def binary_search_beta_torch(perplexity, k_neighbors_distances, tol=1e-5, max_iter=50):
    n = k_neighbors_distances.size(0)  # バッチサイズ
    beta = torch.full((n,), 2.0, device=k_neighbors_distances.device)  # 各データポイントに対して初期betaを設定
    beta_min = torch.full((n,), -float('inf'), device=k_neighbors_distances.device)
    beta_max = torch.full((n,), float('inf'), device=k_neighbors_distances.device)

    for iter in range(max_iter):
        P = torch.exp(-k_neighbors_distances * beta[:, None])
        sum_P = P.sum(dim=1)

        mask_zero = sum_P == 0
        sum_P[mask_zero] = 1e-8
        P /= sum_P[:, None]

        H = -(P * torch.log(P + 1e-8)).sum(dim=1)
        Hdiff = H - torch.log(torch.tensor(perplexity, device=k_neighbors_distances.device))

        mask_positive = Hdiff > 0
        mask_negative = ~mask_positive

        beta_min = torch.where(mask_positive, beta, beta_min)
        beta_max = torch.where(mask_negative, beta, beta_max)

        beta_update_positive = torch.where(beta_max == float('inf'), beta * 2.0, (beta + beta_max) / 2.0)
        beta_update_negative = torch.where(beta_min == -float('inf'), beta / 2.0, (beta + beta_min) / 2.0)

        beta = torch.where(mask_positive, beta_update_positive, beta)
        beta = torch.where(mask_negative, beta_update_negative, beta)

        if torch.all(torch.abs(Hdiff) <= tol):
            break

    return beta

def get_k_neighbors_distances2(X, sample_size, num_neighbors, cpd_distances, restore_tensor):
    print("Get k-neighbors..")
    n = X.shape[0]
    sample_size =  U.closest_divisor(n, sample_size)
    k_neighbors_distances_all = torch.zeros((n, num_neighbors), device=X.device)
    for i in tqdm(range(0, n, sample_size)):
        sample_X = X[i:i + sample_size]
        distance_matrix = get_distance_matrix_of_batch_optimized2(X, sample_X, cpd_distances, restore_tensor)
        distance_matrix = torch.where(distance_matrix == 0.0, distance_matrix.max(), distance_matrix)
        sorted_distances, _ = torch.sort(distance_matrix, dim=1)
        k_neighbors_distances_all[i:i + sample_size, :] = sorted_distances[:, :num_neighbors].cpu()
    k_neighbors_distances_all, _ = torch.sort(k_neighbors_distances_all, dim=1)
    return k_neighbors_distances_all.cpu().numpy()

def get_k_neighbors_and_compute_beta(X, sample_size, num_neighbors, cpd_distances, restore_tensor, perplexity):
    print("Get k-neighbors and compute beta..")
    n = X.shape[0]
    sample_size = U.closest_divisor(n, sample_size)
    precomputed_beta_all = []

    for i in tqdm(range(0, n, sample_size)):
        sample_X = X[i:i + sample_size]
        distance_matrix = get_distance_matrix_of_batch_optimized2(X, sample_X, cpd_distances, restore_tensor)
        distance_matrix = torch.where(distance_matrix == 0.0, distance_matrix.max(), distance_matrix)
        sorted_distances, _ = torch.sort(distance_matrix, dim=1)
        # Compute beta for this batch
        #precomputed_beta = binary_search_beta_torch(perplexity, sorted_distances[:, :num_neighbors])
        precomputed_beta = binary_search_beta(perplexity, sorted_distances[:, :num_neighbors].cpu().numpy())
        #print(precomputed_beta.shape)
        precomputed_beta_all.append(precomputed_beta)

    # Concatenate all beta values
    precomputed_beta_all = torch.cat(precomputed_beta_all, dim=0)
    return precomputed_beta_all.numpy()

def compute_beta(df, perplexity, num_neighbors, cpd_distances, restore_tensor):
    X = df.values  #DataFrameをNumPy配列に変換
    nbrs = get_k_neighbors_distances2(torch.Tensor(X).cuda(), len(df)//100000, num_neighbors, cpd_distances, restore_tensor)
    print("Compute beta..")
    # ここで一度に全てのbetaを計算
    precomputed_beta = binary_search_beta(perplexity, nbrs)
    return precomputed_beta

def compute_beta2(df, perplexity, num_neighbors, cpd_distances, restore_tensor):
    X = df.values  # Convert DataFrame to NumPy array
    precomputed_beta = get_k_neighbors_and_compute_beta(torch.Tensor(X).cuda(), len(df)//1000, num_neighbors, cpd_distances, restore_tensor, perplexity)
    print("Beta computation completed.")
    print(precomputed_beta.shape)
    return precomputed_beta

def js_divergence(P, Q, epsilon=1e-10):
    P = torch.clamp(P, min=epsilon)
    Q = torch.clamp(Q, min=epsilon)
    M = 0.5 * (P + Q)
    kl_pm = F.kl_div(torch.log(M), P, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(M), Q, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)

def rank_func1(distance):
    return torch.stack([torch.sum(torch.relu((torch.relu(distance - d)*10000)+1)) for d in distance]).reshape(1,-1)

def rank_1d(distance):
    return torchsort.soft_rank(distance.view(1, -1)*10000)

def rank_2d(distance_matrix):
    return torchsort.soft_rank(distance_matrix*10000)

def spearman_loss(pred, target):
    # 行ごとにsoft rankを適用
    pred_rank = torchsort.soft_rank((pred)*10000)
    target_rank = torchsort.soft_rank((target)*10000)

    # 平均を引いて、ノルムで割る
    pred_rank = pred_rank - pred_rank.mean(dim=1, keepdim=True)
    target_rank = target_rank - target_rank.mean(dim=1, keepdim=True)
    pred_rank = pred_rank / pred_rank.norm(dim=1, keepdim=True)
    target_rank = target_rank / target_rank.norm(dim=1, keepdim=True)

    # 行ごとに相関を計算
    corr = (pred_rank * target_rank).sum(dim=1)

    # 相関係数の平均を計算
    mean_corr = corr.mean()

    # 損失は1から相関係数の平均を引いたもので、正の相関が高ければ低い損失となる
    loss = 1 - mean_corr
    return loss
