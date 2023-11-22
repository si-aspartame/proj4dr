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
                div = np.sqrt(jensenshannon(cpd1, cpd2))#

            if not div >= 0.0:
                print(cpd1)
                print(cpd2)
                div = 0.0
                raise ValueError

            cpd_distances[num_c_A, num_c_B, unique_A1, unique_A2] = div
            
    return cpd_distances

def get_distance_matrix_of_batch_optimized(X, cpd_distances, restore_tensor):
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