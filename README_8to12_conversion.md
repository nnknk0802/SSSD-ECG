# 8誘導から12誘導ECGへの変換ツール

SSSD-ECGで生成した8誘導ECGデータを標準的な12誘導ECGに変換するスタンドアロンスクリプトです。

## 概要

このツールは、Einthoven's lawとGoldberger's lawを使用して、8誘導ECGデータから残りの4誘導を計算します。

### 入力: 8誘導

1. Lead I
2. V1 (chest lead)
3. V2 (chest lead)
4. V3 (chest lead)
5. V4 (chest lead)
6. V5 (chest lead)
7. V6 (chest lead)
8. aVF

### 出力: 12誘導

1. Lead I (保持)
2. Lead II (計算: 0.5 × Lead I + aVF)
3. V1 (保持)
4. V2 (保持)
5. V3 (保持)
6. V4 (保持)
7. V5 (保持)
8. V6 (保持)
9. Lead III (計算: -0.5 × Lead I + aVF)
10. aVR (計算: -0.75 × Lead I - 0.5 × aVF)
11. aVL (計算: 0.75 × Lead I - 0.5 × aVF)
12. aVF (保持)

## 使用方法

### コマンドラインから使用

```bash
# 基本的な使用法
python convert_8_to_12_leads.py input_8leads.npy output_12leads.npy

# 詳細な情報を表示
python convert_8_to_12_leads.py input_8leads.npy output_12leads.npy --verbose
```

### Pythonコードから使用

```python
import numpy as np
from convert_8_to_12_leads import convert_8_to_12_leads

# Numpy配列を使用
ecg_8_leads = np.load('input_8leads.npy')  # Shape: (batch, 8, length) or (8, length)
ecg_12_leads = convert_8_to_12_leads(ecg_8_leads)
np.save('output_12leads.npy', ecg_12_leads)

# PyTorchテンソルを使用
import torch
ecg_8_tensor = torch.randn(32, 8, 1000)  # バッチサイズ32, 8誘導, 1000サンプル
ecg_12_tensor = convert_8_to_12_leads(ecg_8_tensor)
print(ecg_12_tensor.shape)  # torch.Size([32, 12, 1000])
```

## ファイル構成

- `convert_8_to_12_leads.py`: メインの変換スクリプト
- `test_conversion.py`: テストスクリプト
- `README_8to12_conversion.md`: このドキュメント

## データ形式

### 入力データ

- **ファイル形式**: `.npy` (Numpy配列)
- **Shape**:
  - 単一サンプル: `(8, length)` 例: `(8, 1000)`
  - バッチ: `(batch_size, 8, length)` 例: `(100, 8, 1000)`
- **データ型**: `float32` または `float64`

### 出力データ

- **ファイル形式**: `.npy` (Numpy配列)
- **Shape**:
  - 単一サンプル: `(12, length)`
  - バッチ: `(batch_size, 12, length)`
- **データ型**: 入力と同じ

## 実行例

### 例1: SSSD-ECGで生成したデータを変換

```bash
# SSSD-ECGで8誘導データを生成（既存の inference.py を使用）
python src/sssd/inference.py -c config/config_SSSD_ECG.json

# 生成された8誘導データを12誘導に変換
python convert_8_to_12_leads.py generated_8leads.npy generated_12leads.npy --verbose
```

### 例2: バッチ処理

```python
import numpy as np
from convert_8_to_12_leads import convert_8_to_12_leads

# 複数のバッチファイルを処理
for i in range(6):
    input_file = f'{i}_samples.npy'
    output_file = f'{i}_samples_12lead.npy'

    ecg_8 = np.load(input_file)
    ecg_12 = convert_8_to_12_leads(ecg_8)
    np.save(output_file, ecg_12)

    print(f'変換完了: {input_file} -> {output_file}')
```

## テスト

テストスクリプトを実行して、変換が正しく動作することを確認できます：

```bash
python test_conversion.py
```

テスト内容:
- 単一サンプル（Numpy配列）の変換
- バッチサンプル（Numpy配列）の変換
- 単一サンプル（Torchテンソル）の変換
- バッチサンプル（Torchテンソル）の変換
- 誘導計算の正確性
- ファイル入出力

## 技術的な詳細

### Einthoven's Triangle

標準肢誘導（I, II, III）はEinthoven's triangleを形成します：

```
Lead I = RA → LA
Lead II = RA → LL
Lead III = LA → LL
```

Einthoven's lawより：
```
Lead II = Lead I + Lead III
```

### Goldberger's Augmented Leads

増強単極肢誘導（aVR, aVL, aVF）は以下のように計算されます：

```
aVR = -(Lead I + Lead II) / 2
aVL = (Lead I - Lead III) / 2
aVF = (Lead II + Lead III) / 2
```

### このツールでの実装

8誘導からの変換式：

```python
Lead II = 0.5 * Lead I + aVF
Lead III = -0.5 * Lead I + aVF
aVR = -0.75 * Lead I - 0.5 * aVF
aVL = 0.75 * Lead I - 0.5 * aVF
```

## 依存関係

- Python >= 3.7
- NumPy >= 1.20.0
- PyTorch >= 1.10.0（オプション、Torchテンソルを使用する場合のみ）

## トラブルシューティング

### エラー: "入力は2D (8, length) または 3D (batch_size, 8, length) である必要があります"

入力データの形状を確認してください：

```python
import numpy as np
data = np.load('input.npy')
print(data.shape)  # Should be (8, length) or (batch, 8, length)
```

### エラー: "8誘導のデータが必要です"

データの誘導数を確認してください。このツールは正確に8誘導を必要とします。

```python
print(data.shape[1])  # Should be 8 for 3D input
print(data.shape[0])  # Should be 8 for 2D input
```

## ライセンス

このツールはSSSD-ECGプロジェクトの一部です。

## 関連ファイル

- `src/sssd/inference.py`: SSSD-ECGでの生成と変換の実装
- `src/sssd/dataset.py`: ECGデータセットのローダー
