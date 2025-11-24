#!/usr/bin/env python3
"""
8誘導から12誘導ECGへの変換スクリプト

このスクリプトは8誘導ECGデータを12誘導に変換します。

入力: 8誘導 (Lead I, V1, V2, V3, V4, V5, V6, aVF)
出力: 12誘導 (Lead I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)

使用方法:
    python convert_8_to_12_leads.py input.npy output.npy

またはPythonから:
    from convert_8_to_12_leads import convert_8_to_12_leads
    leads_12 = convert_8_to_12_leads(leads_8)
"""

import argparse
import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # For runtime when torch is not available


def convert_8_to_12_leads(ecg_8_leads: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
    """
    8誘導ECGデータを12誘導に変換します。

    8誘導の構成:
        - Index 0: Lead I
        - Index 1-6: V1-V6 (chest leads)
        - Index 7: aVF

    12誘導の構成:
        - Index 0: Lead I
        - Index 1: Lead II (計算)
        - Index 2-7: V1-V6 (chest leads)
        - Index 8: Lead III (計算)
        - Index 9: aVR (計算)
        - Index 10: aVL (計算)
        - Index 11: aVF

    Parameters:
    -----------
    ecg_8_leads : np.ndarray or torch.Tensor
        8誘導ECGデータ
        Shape: (batch_size, 8, length) または (8, length)

    Returns:
    --------
    ecg_12_leads : np.ndarray or torch.Tensor
        12誘導ECGデータ (入力と同じ型)
        Shape: (batch_size, 12, length) または (12, length)

    Notes:
    ------
    Einthoven's lawとGoldberger's lawを使用した計算:
    - Lead II = 0.5 * Lead I + aVF
    - Lead III = -0.5 * Lead I + aVF
    - aVR = -0.75 * Lead I - 0.5 * aVF
    - aVL = 0.75 * Lead I - 0.5 * aVF
    """
    # 入力の型を保存
    is_numpy = isinstance(ecg_8_leads, np.ndarray)

    # PyTorchテンソルの場合はnumpyに変換
    if not is_numpy:
        if not HAS_TORCH:
            raise ImportError("PyTorchテンソルを使用するにはPyTorchが必要です。pip install torchでインストールしてください。")
        # PyTorchテンソルをnumpy配列に変換
        data = ecg_8_leads.detach().cpu().numpy() if hasattr(ecg_8_leads, 'detach') else ecg_8_leads.numpy()
    else:
        data = ecg_8_leads

    # 2D入力の場合は3Dに変換 (batch_size=1を追加)
    squeeze_output = False
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        squeeze_output = True

    # 入力チェック
    if data.ndim != 3:
        raise ValueError(f"入力は2D (8, length) または 3D (batch_size, 8, length) である必要があります。実際: {data.shape}")

    if data.shape[1] != 8:
        raise ValueError(f"8誘導が必要です。実際の誘導数: {data.shape[1]}")

    # 各誘導を抽出
    lead_I = data[:, 0:1, :]      # Lead I
    chest_leads = data[:, 1:7, :] # V1-V6
    aVF = data[:, 7:8, :]         # aVF

    # Einthoven's lawとGoldberger's lawを使用して残りの誘導を計算
    lead_II = (0.5 * lead_I) + aVF
    lead_III = -(0.5 * lead_I) + aVF
    aVR = -(0.75 * lead_I) - (0.5 * aVF)
    aVL = (0.75 * lead_I) - (0.5 * aVF)

    # 12誘導の順序で結合
    # Lead I, II, III, aVR, aVL, aVF, V1-V6
    leads_12 = np.concatenate([
        lead_I,      # Index 0: Lead I
        lead_II,     # Index 1: Lead II
        chest_leads, # Index 2-7: V1-V6
        lead_III,    # Index 8: Lead III
        aVR,         # Index 9: aVR
        aVL,         # Index 10: aVL
        aVF          # Index 11: aVF
    ], axis=1)

    # 元の次元に戻す
    if squeeze_output:
        leads_12 = leads_12.squeeze(0)

    # PyTorchテンソルで入力された場合はPyTorchテンソルで返す
    if not is_numpy:
        if not HAS_TORCH:
            raise ImportError("PyTorchテンソルを返すにはPyTorchが必要です。")
        leads_12 = torch.from_numpy(leads_12)

    return leads_12


def main():
    """コマンドラインから実行する場合のメイン関数"""
    parser = argparse.ArgumentParser(
        description='8誘導ECGデータを12誘導に変換します',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python convert_8_to_12_leads.py input_8leads.npy output_12leads.npy
  python convert_8_to_12_leads.py input_8leads.npy output_12leads.npy --verbose

入力ファイル形式:
  - .npy形式のNumpy配列
  - Shape: (num_samples, 8, length) または (8, length)
  - 8誘導: Lead I, V1, V2, V3, V4, V5, V6, aVF

出力ファイル形式:
  - .npy形式のNumpy配列
  - Shape: (num_samples, 12, length) または (12, length)
  - 12誘導: Lead I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='入力ファイルパス (.npy形式の8誘導ECGデータ)'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='出力ファイルパス (変換後の12誘導ECGデータを保存)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細な情報を表示'
    )

    args = parser.parse_args()

    # 入力ファイルを読み込み
    try:
        if args.verbose:
            print(f"入力ファイルを読み込み中: {args.input_file}")
        ecg_8_leads = np.load(args.input_file)
        if args.verbose:
            print(f"入力データのshape: {ecg_8_leads.shape}")
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1
    except Exception as e:
        print(f"エラー: 入力ファイルの読み込みに失敗しました: {e}")
        return 1

    # 入力の検証
    if ecg_8_leads.ndim not in [2, 3]:
        print(f"エラー: 入力データは2次元 (8, length) または3次元 (batch, 8, length) である必要があります")
        print(f"実際のshape: {ecg_8_leads.shape}")
        return 1

    lead_dim = 1 if ecg_8_leads.ndim == 3 else 0
    if ecg_8_leads.shape[lead_dim] != 8:
        print(f"エラー: 8誘導のデータが必要です")
        print(f"実際の誘導数: {ecg_8_leads.shape[lead_dim]}")
        return 1

    # 変換実行
    try:
        if args.verbose:
            print("8誘導から12誘導への変換を実行中...")
        ecg_12_leads = convert_8_to_12_leads(ecg_8_leads)
        if args.verbose:
            print(f"出力データのshape: {ecg_12_leads.shape}")
    except Exception as e:
        print(f"エラー: 変換に失敗しました: {e}")
        return 1

    # 出力ファイルに保存
    try:
        if args.verbose:
            print(f"出力ファイルに保存中: {args.output_file}")
        np.save(args.output_file, ecg_12_leads)
        print(f"変換完了: {args.output_file}")
        if args.verbose:
            print(f"  入力: {ecg_8_leads.shape} -> 出力: {ecg_12_leads.shape}")
    except Exception as e:
        print(f"エラー: 出力ファイルの保存に失敗しました: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
