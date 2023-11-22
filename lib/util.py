import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset, TensorDataset
import torch

def loading_data(data_name):
    print(f"loading data --> {data_name}")
    if data_name == "world_happiness_report":
        df = pd.read_csv("data/world_happiness_report/2015.csv", index_col=None)
        df["Year"] = np.repeat(2015, len(df))
        for n in ["2016", "2017", "2018", "2019"]:
            temp_df = pd.read_csv(f"data/world_happiness_report/{n}.csv", index_col=None)
            temp_df["Year"] = np.repeat(n, len(temp_df))
            df = pd.concat([df, temp_df])
    elif data_name == "wine":
        df = pd.read_csv("data/wine/wine.csv", index_col=None)
    elif data_name == "data_science_salary":
        df = pd.read_csv("data/data_science_salary/Data Science Salary 2021 to 2023.csv", index_col=None)
    elif data_name == "global_electricity_data":
        df = pd.read_csv("data/global_electricity_statistics/Global Electricity Statistics.csv", index_col=None)
    elif data_name == "adult":
        df = pd.read_csv("data/adult/adult.csv", index_col=None)
    elif data_name == "creditcard":
        df = pd.read_csv("data/creditcard/creditcard_2023.csv", index_col=None)
    elif data_name == "mnist":
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        df = pd.DataFrame(mnist["data"])
        df["target"] = pd.DataFrame(mnist["target"].astype(float)).values
    print(f"columns_size:{len(df.columns)}")
    if not df.isna().mean().sum != 0.0:
        raise ValueError
    return df

def encode_object_column(df):
    for col in df.select_dtypes(include=['object']).columns:
        series = df[col]
        # 列の各要素を数値に変換し試み、その後数値かどうかを確認
        is_digit = pd.to_numeric(series, errors='coerce').notna()
        # 数字の割合を計算
        digit_ratio = is_digit.mean()
        # 半分以上が数字の場合にのみ変換を実行
        if digit_ratio > 0.9:
            df[col] = pd.to_numeric(series, errors='coerce').fillna(0)
            df[col] = df[col].astype(float)
    return df

def equal_frequency_binning(df, bin_size):
    for column in df.select_dtypes(include=['float', 'int']):
        unique_values = len(df[column].unique())
        if unique_values >= bin_size:  # 2より大きい場合のみビニングを適用
            df[column] = pd.qcut(df[column], q=bin_size, labels=False, duplicates='drop')
    return df

def label_by_counts(df):
    for c in df.columns:
        column = df[c]
        value_counts = column.value_counts().sort_values(ascending=False).reset_index()
        value_counts.columns = ['value', 'count']

        custom_encoding = {}
        for n, val in enumerate(value_counts['value']):
            custom_encoding[val] = n

        df[c] = column.map(custom_encoding)
    return df

def replace_low_frequency_values(df, threshold=2):
    for col in df.columns:
        value_counts = df[col].value_counts()
        to_replace = value_counts[value_counts <= threshold].index
        df[col] = df[col].replace(to_replace, -1)
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def normalize_all_columns(df):
    # 分母が0になる列を削除
    min_vals = df.min()
    max_vals = df.max()
    diff = max_vals - min_vals
    valid_columns = diff[diff != 0].index
    
    # 99%以上が同じ値の列を削除
    n_rows = len(df)
    for col in valid_columns:
        most_frequent_count = df[col].value_counts().iloc[0]
        if (most_frequent_count / n_rows) >= 0.99:
            valid_columns = valid_columns.drop(col)
    
    df = df[valid_columns]
    
    # 正規化
    df = 2 * ((df - min_vals[valid_columns]) / diff[valid_columns]) - 1
    
    # 列数を取得
    n_columns = len(df.columns)
    
    # restore_tensor の形状を [n_columns, 2] とする
    restore_tensor = torch.zeros((n_columns, 2), dtype=torch.float32)
    
    for i, col in enumerate(df.columns):
        restore_tensor[i, 0] = min_vals[col]
        restore_tensor[i, 1] = max_vals[col]
    
    return df, restore_tensor

def dataframe_to_dataset(df, label_column=None):
    if label_column is not None:
        # If label_column is specified, separate the dataframe into data and labels
        labels = df[label_column].values
        data = df.drop(columns=[label_column]).values
    else:
        # If label_column is not specified, use sequential numbers as labels
        labels = np.arange(len(df))
        data = df.values
    
    # Convert data and labels to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)
    
    return dataset

def get_tensor_memory_size(tensor):
    # テンソルのdtypeから1要素あたりのバイト数を取得
    bytes_per_element = tensor.element_size()
    
    # テンソルの全要素数を取得
    num_elements = tensor.numel()
    
    # テンソルのメモリ使用量（バイト単位）を計算
    total_bytes = bytes_per_element * num_elements
    
    return total_bytes / (1024 ** 3)

def closest_divisor(x, y):
    for z in range(y - 1, 0, -1):
        if x % z == 0:
            return z
    return None  # Return None if no such divisor is found
