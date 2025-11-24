# 8誘導から12誘導ECGへの変換

## 概要

このドキュメントでは、8誘導ECGデータ（I, II, V1-V6）から12誘導ECGデータへの変換方法について説明します。

## ECG誘導の基礎

### 12誘導ECGの構成

標準的な12誘導ECGは以下の誘導で構成されています：

1. **肢誘導 (Limb Leads)**
   - 双極誘導：I, II, III
   - 単極誘導：aVR, aVL, aVF

2. **胸部誘導 (Chest Leads)**
   - V1, V2, V3, V4, V5, V6

### 8誘導ECGの構成

PTB-XLデータセットでデフォルトとして使用される8誘導は以下の通りです：

- I, II, V1, V2, V3, V4, V5, V6

つまり、III, aVR, aVL, aVFが欠落しています。

## 導出誘導の計算

欠落している4つの肢誘導は、アイントホーフェンの三角形の原理に基づいて導出できます：

### 数式

```
Lead III = Lead II - Lead I
aVR = -(Lead I + Lead II) / 2
aVL = Lead I - Lead II / 2
aVF = Lead II - Lead I / 2
```

### アイントホーフェンの三角形

```
        aVR
         |
         |
    I --------> II
         |
        III
```

## 使用方法

### 1. コマンドラインツールとして使用

```bash
# 基本的な使用方法
python src/sssd/convert_8_to_12_lead.py --input data_8lead.npy --output data_12lead.npy

# 詳細情報を表示
python src/sssd/convert_8_to_12_lead.py -i data_8lead.npy -o data_12lead.npy -v

# ヘルプを表示
python src/sssd/convert_8_to_12_lead.py --help
```

### 2. Pythonライブラリとして使用

```python
import numpy as np
from src.sssd.convert_8_to_12_lead import convert_8_to_12_lead

# 8誘導データを読み込む
data_8lead = np.load('data_8lead.npy')  # shape: (num_samples, 8, signal_length)

# 12誘導に変換
data_12lead = convert_8_to_12_lead(data_8lead)  # shape: (num_samples, 12, signal_length)

# 結果を保存
np.save('data_12lead.npy', data_12lead)
```

### 3. 単一サンプルの変換

```python
# 単一サンプルの場合
single_sample_8lead = np.load('single_ecg.npy')  # shape: (8, 1000)
single_sample_12lead = convert_8_to_12_lead(single_sample_8lead)  # shape: (12, 1000)
```

## データフォーマット

### 入力データ

- **形状**: `(num_samples, 8, signal_length)` または `(8, signal_length)`
- **誘導順序**: I, II, V1, V2, V3, V4, V5, V6
- **データタイプ**: `float32` または `float64`

### 出力データ

- **形状**: `(num_samples, 12, signal_length)` または `(12, signal_length)`
- **誘導順序**: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- **データタイプ**: 入力と同じ

## サンプルコード

### 例1: バッチ変換

```python
import numpy as np
from src.sssd.convert_8_to_12_lead import convert_8_to_12_lead

# 複数のECGサンプルを変換
data_8lead = np.load('training_data_8lead.npy')  # (1000, 8, 1000)
data_12lead = convert_8_to_12_lead(data_8lead)    # (1000, 12, 1000)

print(f"Input shape:  {data_8lead.shape}")   # (1000, 8, 1000)
print(f"Output shape: {data_12lead.shape}")  # (1000, 12, 1000)
```

### 例2: PTB-XLデータセットとの統合

```python
import numpy as np
from src.sssd.dataset import PTBXLDataset
from src.sssd.convert_8_to_12_lead import convert_8_to_12_lead

# PTB-XL 8誘導データを読み込む
dataset = PTBXLDataset(
    data_path='ptbxl_data.npy',
    labels_path='ptbxl_labels.npy',
    use_8_leads=True  # 8誘導モード
)

# データローダーから8誘導データを取得
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)

for ecg_8lead, labels in loader:
    # PyTorchテンソルをnumpy配列に変換
    ecg_8lead_np = ecg_8lead.numpy()

    # 12誘導に変換
    ecg_12lead_np = convert_8_to_12_lead(ecg_8lead_np)

    # 必要に応じてテンソルに戻す
    import torch
    ecg_12lead = torch.from_numpy(ecg_12lead_np)

    # 処理を続行...
```

### 例3: テストデータの生成と検証

完全な使用例については、`example_8_to_12_lead_conversion.py`を参照してください：

```bash
python src/sssd/example_8_to_12_lead_conversion.py
```

このスクリプトは以下の例を含みます：
- 単一サンプルの変換
- バッチ変換
- データの保存と読み込み
- PTB-XL形式との互換性
- 変換結果の検証

## 検証方法

変換が正しく行われたことを確認するには：

```python
def validate_conversion(data_8lead, data_12lead):
    # 元の誘導が保持されているか確認
    assert np.allclose(data_8lead[:, 0, :], data_12lead[:, 0, :])  # Lead I
    assert np.allclose(data_8lead[:, 1, :], data_12lead[:, 1, :])  # Lead II

    # 導出誘導が正しいか確認
    lead_I = data_12lead[:, 0, :]
    lead_II = data_12lead[:, 1, :]
    lead_III = data_12lead[:, 2, :]

    assert np.allclose(lead_III, lead_II - lead_I)

    print("✓ 変換が正しく行われました")
```

## 注意事項

1. **誘導の順序**: 入力データの誘導順序が正しいことを確認してください（I, II, V1-V6）
2. **データ形状**: 入力データは `(num_samples, 8, signal_length)` または `(8, signal_length)` の形状である必要があります
3. **信号長**: 信号長は任意ですが、PTB-XLの場合は通常1000サンプル（100Hzで10秒）です
4. **データタイプ**: `float32`または`float64`を推奨します

## トラブルシューティング

### エラー: "Expected 8 leads, got X"

入力データの誘導数が8でない場合に発生します。データの形状を確認してください：

```python
print(data.shape)  # (num_samples, num_leads, signal_length)
```

### PTB-XL形式のデータ

PTB-XLデータが `(num_samples, signal_length, num_leads)` の形式の場合、転置が必要です：

```python
data_8lead = np.transpose(data_ptbxl, (0, 2, 1))
```

## 参考文献

1. アイントホーフェンの三角形と誘導理論
2. PTB-XLデータセット: https://physionet.org/content/ptb-xl/
3. 標準12誘導ECGの解釈

## ライセンス

このコードはSSSD-ECGプロジェクトの一部です。
