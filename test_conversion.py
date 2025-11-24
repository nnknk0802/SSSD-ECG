#!/usr/bin/env python3
"""
8誘導から12誘導への変換スクリプトのテスト
"""

import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Skipping PyTorch tests.")

from convert_8_to_12_leads import convert_8_to_12_leads


def test_numpy_single_sample():
    """単一サンプルのNumpy配列をテスト"""
    print("テスト1: 単一サンプル (Numpy配列)")

    # 8誘導のダミーデータを生成 (8, 1000)
    ecg_8 = np.random.randn(8, 1000).astype(np.float32)
    print(f"  入力shape: {ecg_8.shape}")

    # 変換
    ecg_12 = convert_8_to_12_leads(ecg_8)
    print(f"  出力shape: {ecg_12.shape}")

    # 検証
    assert ecg_12.shape == (12, 1000), f"期待されるshape: (12, 1000), 実際: {ecg_12.shape}"
    assert isinstance(ecg_12, np.ndarray), "出力はNumpy配列であるべき"

    # Lead Iが保持されているか確認
    assert np.allclose(ecg_12[0], ecg_8[0]), "Lead Iが正しく保持されていません"

    # Chest leads (V1-V6)が保持されているか確認
    assert np.allclose(ecg_12[2:8], ecg_8[1:7]), "Chest leadsが正しく保持されていません"

    # aVFが保持されているか確認
    assert np.allclose(ecg_12[11], ecg_8[7]), "aVFが正しく保持されていません"

    print("  ✓ テスト成功")
    return True


def test_numpy_batch():
    """バッチのNumpy配列をテスト"""
    print("\nテスト2: バッチサンプル (Numpy配列)")

    # 8誘導のダミーデータを生成 (32, 8, 1000)
    batch_size = 32
    ecg_8 = np.random.randn(batch_size, 8, 1000).astype(np.float32)
    print(f"  入力shape: {ecg_8.shape}")

    # 変換
    ecg_12 = convert_8_to_12_leads(ecg_8)
    print(f"  出力shape: {ecg_12.shape}")

    # 検証
    assert ecg_12.shape == (batch_size, 12, 1000), f"期待されるshape: ({batch_size}, 12, 1000), 実際: {ecg_12.shape}"
    assert isinstance(ecg_12, np.ndarray), "出力はNumpy配列であるべき"

    print("  ✓ テスト成功")
    return True


def test_torch_single_sample():
    """単一サンプルのTorchテンソルをテスト"""
    if not HAS_TORCH:
        print("\nテスト3: 単一サンプル (Torchテンソル) - スキップ（PyTorch未インストール）")
        return True

    print("\nテスト3: 単一サンプル (Torchテンソル)")

    # 8誘導のダミーデータを生成 (8, 1000)
    ecg_8 = torch.randn(8, 1000)
    print(f"  入力shape: {ecg_8.shape}")

    # 変換
    ecg_12 = convert_8_to_12_leads(ecg_8)
    print(f"  出力shape: {ecg_12.shape}")

    # 検証
    assert ecg_12.shape == (12, 1000), f"期待されるshape: (12, 1000), 実際: {ecg_12.shape}"
    assert isinstance(ecg_12, torch.Tensor), "出力はTorchテンソルであるべき"

    print("  ✓ テスト成功")
    return True


def test_torch_batch():
    """バッチのTorchテンソルをテスト"""
    if not HAS_TORCH:
        print("\nテスト4: バッチサンプル (Torchテンソル) - スキップ（PyTorch未インストール）")
        return True

    print("\nテスト4: バッチサンプル (Torchテンソル)")

    # 8誘導のダミーデータを生成 (32, 8, 1000)
    batch_size = 32
    ecg_8 = torch.randn(batch_size, 8, 1000)
    print(f"  入力shape: {ecg_8.shape}")

    # 変換
    ecg_12 = convert_8_to_12_leads(ecg_8)
    print(f"  出力shape: {ecg_12.shape}")

    # 検証
    assert ecg_12.shape == (batch_size, 12, 1000), f"期待されるshape: ({batch_size}, 12, 1000), 実際: {ecg_12.shape}"
    assert isinstance(ecg_12, torch.Tensor), "出力はTorchテンソルであるべき"

    print("  ✓ テスト成功")
    return True


def test_lead_calculations():
    """誘導の計算が正しいか詳細にテスト"""
    print("\nテスト5: 誘導計算の正確性")

    # 簡単な値で8誘導データを作成
    ecg_8 = np.zeros((8, 100), dtype=np.float32)

    # Lead I = 1.0
    ecg_8[0, :] = 1.0
    # V1-V6 = 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    for i in range(6):
        ecg_8[i+1, :] = float(i + 2)
    # aVF = 8.0
    ecg_8[7, :] = 8.0

    # 変換
    ecg_12 = convert_8_to_12_leads(ecg_8)

    # 期待される値を計算
    lead_I = 1.0
    aVF = 8.0
    expected_lead_II = 0.5 * lead_I + aVF  # 0.5 + 8.0 = 8.5
    expected_lead_III = -0.5 * lead_I + aVF  # -0.5 + 8.0 = 7.5
    expected_aVR = -0.75 * lead_I - 0.5 * aVF  # -0.75 - 4.0 = -4.75
    expected_aVL = 0.75 * lead_I - 0.5 * aVF  # 0.75 - 4.0 = -3.25

    print(f"  Lead I: {ecg_12[0, 0]:.2f} (期待値: {lead_I:.2f})")
    print(f"  Lead II: {ecg_12[1, 0]:.2f} (期待値: {expected_lead_II:.2f})")
    print(f"  Lead III: {ecg_12[8, 0]:.2f} (期待値: {expected_lead_III:.2f})")
    print(f"  aVR: {ecg_12[9, 0]:.2f} (期待値: {expected_aVR:.2f})")
    print(f"  aVL: {ecg_12[10, 0]:.2f} (期待値: {expected_aVL:.2f})")
    print(f"  aVF: {ecg_12[11, 0]:.2f} (期待値: {aVF:.2f})")

    # 検証
    assert np.allclose(ecg_12[0, :], lead_I), "Lead Iが正しくありません"
    assert np.allclose(ecg_12[1, :], expected_lead_II), "Lead IIが正しくありません"
    assert np.allclose(ecg_12[8, :], expected_lead_III), "Lead IIIが正しくありません"
    assert np.allclose(ecg_12[9, :], expected_aVR), "aVRが正しくありません"
    assert np.allclose(ecg_12[10, :], expected_aVL), "aVLが正しくありません"
    assert np.allclose(ecg_12[11, :], aVF), "aVFが正しくありません"

    # Chest leadsの確認
    for i in range(6):
        expected_val = float(i + 2)
        assert np.allclose(ecg_12[i+2, :], expected_val), f"V{i+1}が正しくありません"
        print(f"  V{i+1}: {ecg_12[i+2, 0]:.2f} (期待値: {expected_val:.2f})")

    print("  ✓ テスト成功")
    return True


def test_file_io():
    """ファイル入出力のテスト"""
    print("\nテスト6: ファイル入出力")

    import tempfile
    import os

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix='_input.npy', delete=False) as tmp_input:
        input_file = tmp_input.name

    with tempfile.NamedTemporaryFile(suffix='_output.npy', delete=False) as tmp_output:
        output_file = tmp_output.name

    try:
        # テストデータを作成して保存
        ecg_8 = np.random.randn(10, 8, 1000).astype(np.float32)
        np.save(input_file, ecg_8)
        print(f"  テストデータを保存: {input_file}")
        print(f"  入力shape: {ecg_8.shape}")

        # 変換関数を使用
        ecg_8_loaded = np.load(input_file)
        ecg_12 = convert_8_to_12_leads(ecg_8_loaded)
        np.save(output_file, ecg_12)
        print(f"  変換データを保存: {output_file}")

        # 保存されたファイルを読み込んで検証
        ecg_12_loaded = np.load(output_file)
        print(f"  出力shape: {ecg_12_loaded.shape}")

        assert ecg_12_loaded.shape == (10, 12, 1000), "出力shapeが正しくありません"
        assert np.allclose(ecg_12, ecg_12_loaded), "保存/読み込みでデータが変わりました"

        print("  ✓ テスト成功")
        return True

    finally:
        # 一時ファイルを削除
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def main():
    """全てのテストを実行"""
    print("=" * 60)
    print("8誘導から12誘導への変換テスト")
    print("=" * 60)

    tests = [
        test_numpy_single_sample,
        test_numpy_batch,
        test_torch_single_sample,
        test_torch_batch,
        test_lead_calculations,
        test_file_io,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ テスト失敗: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"テスト結果: {passed}個成功, {failed}個失敗")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
