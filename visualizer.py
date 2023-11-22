#%%
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import lib.util as U
from sklearn.preprocessing import LabelEncoder

# original.csvから元のデータフレームを読み込み
#%%
original_df = pd.read_csv('original.csv', index_col=0)
#%%
print(original_df.dtypes)

# array.csvからt-SNE用のデータを読み込み
tsne_data = pd.read_csv('encoded.csv', header=None)

# t-SNEで次元削減
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(tsne_data)

# %%
# カラムの数
num_columns = len(original_df.columns)

# グリッドのサイズ
grid_rows = num_columns // 4 + 1

# プロットを並べるフィギュアを作成
fig, axes = plt.subplots(grid_rows, 4, figsize=(30, 8 * grid_rows))

# カラムごとにプロット
for i, column in enumerate(original_df.columns):
    row = i // 4
    col = i % 4

    # カテゴリカルデータを数値に変換
    if original_df[column].dtype == 'object':
        cmap = plt.cm.tab10
        label_encoder = LabelEncoder()

        # トップ10のカテゴリを選択
        top_categories = original_df[column].value_counts().head(10).index
        reduced_df = original_df[original_df[column].isin(top_categories)]
        encoded = label_encoder.fit_transform(reduced_df[column])
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

        scatter = axes[row, col].scatter(tsne_results[reduced_df.index, 0], tsne_results[reduced_df.index, 1], c=encoded, cmap=cmap, s=5)
        axes[row, col].set_title(f'{column} (Categorical)')

        # 凡例のラベルを手動で設定
        legend_labels = {v: k for k, v in label_mapping.items()}  # ラベルと数値のマッピングを反転
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], 
                              markersize=5, markerfacecolor=cmap(i/len(label_mapping))) for i in label_mapping.values()]
        axes[row, col].legend(handles=handles, title=column)
    else:
        cmap = plt.cm.viridis

        scatter = axes[row, col].scatter(tsne_results[:, 0], tsne_results[:, 1], c=original_df[column], cmap=cmap, s=5)
        axes[row, col].set_title(f'{column} (Numerical)')
        fig.colorbar(scatter, ax=axes[row, col], label=column)

