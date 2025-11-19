# SSSD-ECG リファクタリング版

このディレクトリには、SSSD-ECGモデルのリファクタリング版が含まれています。データローディングとモデルの実装を分離し、使いやすいインターフェースを提供します。

## 主な変更点

### 1. クリーンなモデルインターフェース

新しい`SSSDECG`ラッパークラスにより、モデルの使用が簡単になりました:

```python
from model_wrapper import SSSDECG

# モデルの初期化
model = SSSDECG(config_path="config/config_SSSD_ECG.json")

# 学習: lossの計算
loss = model(x, y)

# 推論: サンプルの生成
pred = model.generate(labels=y, num_samples=10)
```

### 2. 分離されたデータローディング

`dataset.py`に専用のDatasetクラスを実装:

- `ECGDataset`: 汎用的なECGデータセットクラス
- `PTBXLDataset`: PTB-XL専用のデータセットクラス
- `create_dataloaders()`: DataLoaderを簡単に作成するユーティリティ関数

### 3. 新しい学習スクリプト

`train_new.py`は新しいインターフェースを使用したクリーンな学習スクリプトです。

## ファイル構成

```
src/sssd/
├── model_wrapper.py          # SSSDECGラッパークラス
├── dataset.py                # データセットクラス
├── train_new.py              # 新しい学習スクリプト
├── example_usage.py          # 使用例
├── README_REFACTORED.md      # このファイル
├── models/
│   └── SSSD_ECG.py          # 元のモデル実装(変更なし)
└── utils/
    └── util.py               # ユーティリティ関数(変更なし)
```

## 使用方法

### 基本的な使い方

```python
import torch
from model_wrapper import SSSDECG
from dataset import ECGDataset
from torch.utils.data import DataLoader

# 1. データセットの作成
dataset = ECGDataset(
    data_path="ptbxl_train_data.npy",
    labels_path="ptbxl_train_labels.npy",
    segment_length=1000
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. モデルの初期化
model = SSSDECG(config_path="config/config_SSSD_ECG.json")

# 3. Optimizerの設定
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 4. 学習ループ
for x, y in dataloader:
    loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
```

### 学習スクリプトの実行

```bash
# デフォルト設定で学習
python train_new.py --config config/config_SSSD_ECG.json

# カスタムパラメータで学習
python train_new.py \
    --config config/config_SSSD_ECG.json \
    --data_path /path/to/data.npy \
    --labels_path /path/to/labels.npy \
    --batch_size 16 \
    --lr 1e-4 \
    --n_iters 50000

# チェックポイントから再開
python train_new.py \
    --config config/config_SSSD_ECG.json \
    --checkpoint checkpoints/40000.pkl
```

### サンプル生成

```python
from model_wrapper import SSSDECG
import torch

# モデルの初期化
model = SSSDECG(config_path="config/config_SSSD_ECG.json")

# チェックポイントの読み込み
model.load_checkpoint("checkpoints/100000.pkl")

# ランダムなラベルでサンプル生成
samples = model.generate(num_samples=10, return_numpy=True)
print(f"Generated shape: {samples.shape}")  # (10, 8, 1000)

# 特定のラベルでサンプル生成
labels = torch.tensor([0, 5, 10, 15, 20])  # クラスインデックス
samples = model.generate(labels=labels, return_numpy=True)
```

## API リファレンス

### SSSDECG クラス

#### `__init__(config_path=None, device=None)`

モデルを初期化します。

**パラメータ:**
- `config_path` (str): 設定ファイルのパス。デフォルト: `"config/config_SSSD_ECG.json"`
- `device` (str): 使用するデバイス。デフォルト: `"cuda"`(利用可能な場合)

#### `forward(x, y)`

学習時のlossを計算します。

**パラメータ:**
- `x` (torch.Tensor): ECG信号, shape=(batch_size, channels, length)
- `y` (torch.Tensor): ラベル, shape=(batch_size, num_classes)または(batch_size,)

**戻り値:**
- loss (torch.Tensor): スカラーのloss値

#### `generate(labels=None, num_samples=1, return_numpy=False)`

ECGサンプルを生成します。

**パラメータ:**
- `labels` (torch.Tensor, optional): 条件付けラベル
- `num_samples` (int): 生成するサンプル数
- `return_numpy` (bool): Trueの場合、numpy配列を返す

**戻り値:**
- samples (torch.Tensor or np.ndarray): 生成されたECG信号

#### `load_checkpoint(checkpoint_path)`

チェックポイントからモデルの重みを読み込みます。

#### `save_checkpoint(checkpoint_path, optimizer=None, epoch=None)`

モデルの重みをチェックポイントに保存します。

### ECGDataset クラス

PyTorch Dataset for ECG data.

**パラメータ:**
- `data_path` (str or np.ndarray): データファイルのパスまたはnumpy配列
- `labels_path` (str or np.ndarray, optional): ラベルファイルのパスまたはnumpy配列
- `transform` (callable, optional): ECG信号に適用する変換
- `target_transform` (callable, optional): ラベルに適用する変換
- `lead_indices` (list, optional): 選択するリードのインデックス
- `segment_length` (int, optional): ECGセグメントの長さ

### create_dataloaders 関数

学習用と検証用のDataLoaderを簡単に作成します。

```python
train_loader, val_loader = create_dataloaders(
    train_data_path="train_data.npy",
    train_labels_path="train_labels.npy",
    val_data_path="val_data.npy",
    val_labels_path="val_labels.npy",
    batch_size=8,
    num_workers=4,
    lead_indices=[0, 1, 6, 7, 8, 9, 10, 11],
    segment_length=1000
)
```

## 使用例

詳細な使用例は`example_usage.py`を参照してください:

```bash
python example_usage.py
```

このスクリプトには以下の例が含まれています:
1. 基本的な学習ループ
2. サンプル生成
3. チェックポイントの保存/読み込み
4. 実データでの学習
5. カスタムDatasetの使用

## 元の実装との互換性

元のモデル実装(`models/SSSD_ECG.py`)と学習スクリプト(`train.py`)は変更されていません。
新しいインターフェースは、これらの既存のコードを内部で使用しています。

## 設定ファイル

`config/config_SSSD_ECG.json`で以下のパラメータを設定できます:

```json
{
  "diffusion_config": {
    "T": 200,              // 拡散ステップ数
    "beta_0": 0.0001,      // ノイズスケジュールの開始値
    "beta_T": 0.02         // ノイズスケジュールの終了値
  },
  "wavenet_config": {
    "in_channels": 8,      // 入力チャンネル数
    "out_channels": 8,     // 出力チャンネル数
    "num_res_layers": 36,  // 残差ブロック数
    "res_channels": 256,   // 残差チャンネル数
    // ... その他のモデルパラメータ
  },
  "train_config": {
    "learning_rate": 2e-4,
    "batch_size": 8,
    "n_iters": 100000,
    // ... その他の学習パラメータ
  }
}
```

## 推奨事項

### メモリ効率化

- 大規模データセットの場合、`num_workers`を増やしてデータローディングを並列化
- バッチサイズをGPUメモリに応じて調整

### 学習の安定化

- 学習率スケジューラの使用を検討
- Gradient clippingの追加を検討

### モニタリング

- TensorBoardやWandBなどのロギングツールの統合を検討
- 定期的な検証セットでの評価

## トラブルシューティング

### CUDA out of memory

```python
# バッチサイズを減らす
train_loader, _ = create_dataloaders(..., batch_size=4)

# または勾配累積を使用
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### データ形状のエラー

データが正しい形状であることを確認:
- ECG data: `(num_samples, channels, length)`
- Labels: `(num_samples,)` または `(num_samples, num_classes)`

## 今後の拡張予定

- [ ] データ拡張のサポート
- [ ] マルチGPU学習のサポート
- [ ] より柔軟な条件付け機能
- [ ] 学習進捗の可視化ツール
