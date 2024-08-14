"""
音声データを読み込み、メルスペクトログラムを計算して特徴量に変換する
モデルを構築し、学習させる
"""

import librosa
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile

def extract_mel_spectrogram(file_path: str, max_length: int) -> np.ndarray:
    """
    音声ファイルからメルスペクトログラムを抽出する関数

    Args:
        file_path (str): 音声ファイルのパス

    Returns:
        np.ndarray: メルスペクトログラムの特徴量
    """
    audio, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    feature = log_mel_spectrogram.flatten()

    # 特徴量の長さを一定にするためにパディング
    if len(feature) < max_length:
        feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
    else:
        feature = feature[:max_length]

    return feature

def load_gender_labels(file_path: str) -> dict:
    """
    gender_f0range.txt ファイルから性別ラベルを読み込む関数

    Args:
        file_path (str): gender_f0range.txt ファイルのパス

    Returns:
        dict: 音声ファイル名をキー、性別ラベルを値とする辞書
    """
    gender_labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            # speaker Male_or_Female minf0[Hz] maxf0[Hz]
            parts = line.strip().split()
            if len(parts) == 4:
                file_id, gender, _, _ = parts
                if gender == 'M':
                    gender_labels[file_id] = 1
                elif gender == 'F':
                    gender_labels[file_id] = 0
    return gender_labels


def create_dataset(directory: str, gender_labels: dict, progress_label: tk.Label) -> pd.DataFrame:
    """
    ディレクトリ内の音声ファイルからデータセットを作成する関数

    Args:
        directory (str): 音声ファイルが保存されているディレクトリのパス
        gender_labels (dict): 音声ファイル名をキー、性別ラベルを値とする辞書
        progress_label (tk.Label): 進捗状況を表示するラベル

    Returns:
        pd.DataFrame: 特徴量とラベルを含むデータフレーム
    """
    features = []
    labels = []
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    processed_files = 0

    for root, dirs, files in os.walk(directory):
        print(root)
        dir_name = root.split('/')[2]
        print(dir_name)
        for file_name in files:
            progress_label.config(text=f"データセット作成中: {processed_files}/{total_files}")
            progress_label.update_idletasks()
            processed_files += 1
            if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                if dir_name in gender_labels:
                    file_path = os.path.join(root, file_name)
                    feature = extract_mel_spectrogram(file_path, 1000)
                    label = gender_labels[dir_name]
                    features.append(feature)
                    labels.append(label)
                
    
    # デバッグ情報を追加
    print(f"Total features extracted: {len(features)}")
    print(f"Total labels extracted: {len(labels)}")
    
    return pd.DataFrame({'feature': features, 'label': labels})

def build_model(input_shape: int, dropout_rate: float) -> tf.keras.Model:
    """
    性別分類モデルを構築する関数

    Args:
        input_shape (int): 入力データの形状
        dropout_rate (float): ドロップアウト率

    Returns:
        tf.keras.Model: 構築されたモデル
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        # 出力層（バイナリクラス分類）
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def unzip_file(zip_path: str, extract_to: str):
    """
    ZIPファイルを展開する関数

    Args:
        zip_path (str): ZIPファイルのパス
        extract_to (str): 展開先のディレクトリ
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def make_model(data_path: str, model_path: str, progress_label: tk.Label):
    """
    モデルを作成する関数

    Args:
        data_path (str): 学習データのZIPファイルのパス
        model_path (str): 保存するモデルファイルのパス
        progress_label (tk.Label): 進捗状況を表示するラベル
    """
    progress_label.config(text="Unzipping data...")
    progress_label.update_idletasks()

    file_name = os.path.basename(data_path)
    file_name = os.path.splitext(file_name)[0]

    # 展開先ディレクトリを定義
    extract_to = 'data/extracted_data'

    # 展開されてないのであれば
    if not os.path.exists(os.path.join(extract_to, file_name)):
        # ZIPファイルを展開
        progress_label.config(text="学習データを解凍中...")
        progress_label.update_idletasks()
        unzip_file(data_path, extract_to)

    extracted_file = os.path.join(extract_to, file_name)

    progress_label.config(text="性別ラベルを読み込み中...")
    progress_label.update_idletasks()

    # 性別ラベルの読み込み
    gender_labels = load_gender_labels(os.path.join(extracted_file, 'gender_f0range.txt'))
    
    # デバッグ情報を追加
    print(f"Total gender labels loaded: {len(gender_labels)}")
    print(gender_labels)

    progress_label.config(text="データセットを作成中...")
    progress_label.update_idletasks()

    # データセットの作成
    dataset = create_dataset(extracted_file, gender_labels, progress_label)
    
    # デバッグ情報を追加
    print(f"Dataset created with {len(dataset)} samples")
    print(dataset.head())

    # データセットが空でないことを確認
    if dataset.empty:
        progress_label.config(text="データセットが空です")
        return

    X = np.array([np.fromstring(feature[1:-1], sep=' ') for feature in dataset['feature']])
    y = dataset['label'].values

    # Xが2次元配列であることを確認
    if X.ndim != 2:
        progress_label.config(text="Feature array is not 2D. Please check the feature extraction process.")
        return

    progress_label.config(text="データを標準化...")
    progress_label.update_idletasks()

    # データの標準化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    progress_label.config(text="データの分割中...")
    progress_label.update_idletasks()

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    progress_label.config(text="モデルの構築中...")
    progress_label.update_idletasks()

    # モデルの構築
    model = build_model(X_train.shape[1], 0.5)

    progress_label.config(text="モデルのトレーニング中...")
    progress_label.update_idletasks()

    # モデルのトレーニング
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    progress_label.config(text="モデルの保存中...")
    progress_label.update_idletasks()

    # モデルの保存
    model.save(model_path)

    progress_label.config(text="モデルのトレーニング完了！")
    progress_label.update_idletasks()