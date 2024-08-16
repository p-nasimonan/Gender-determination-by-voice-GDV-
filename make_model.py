"""
音声データを読み込み、メルスペクトログラムを計算して特徴量に変換する
モデルを構築し、学習させる
"""

import librosa
import numpy as np
import pandas as pd
import os
import tkinter as tk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile
import requests

def download_data(url: str, save_path: str):
    """
    データをダウンロードする関数
    """
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def extract_log_mel_spectrogram(file_path: str, max_length: int) -> np.ndarray:
    """
    音声ファイルからメルスペクトログラムを抽出する関数

    Args:
        file_path (str): 音声ファイルのパス
        max_length (int): 特徴量の最大長さ

    Returns:
        np.ndarray: メルスペクトログラムの特徴量
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # フラット化してリストに変換
        feature = log_mel_spectrogram.flatten()
        
        # 特徴量の長さを一定にするためにパディング
        if len(feature) < max_length:
            feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
        else:
            feature = feature[:max_length]
        
        return feature
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])

def load_gender_labels(file_path: str) -> dict:
    """
    gender_f0range.txt ファイルから性別ラベルを読み込む関数

    Args:
        file_path (str): gender_f0range.txt ファイルのパス

    Returns:
        dict: 音声ファイル名をキー、性別ラベルを値とする辞書
    """
    gender_labels = {}
    try:
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
    except Exception as e:
        print(f"Error reading gender labels: {e}")
    return gender_labels


def create_dataset(directory: str, gender_labels: dict, progress_label: tk.Label, max_length: int) -> pd.DataFrame:
    """
    ディレクトリ内の音声ファイルからデータセットを作成する関数

    Args:
        directory (str): 音声ファイルが保存されているディレクトリのパス
        gender_labels (dict): 音声ファイル名をキー、性別ラベルを値とする辞書
        progress_label (tk.Label): 進捗状況を表示するラベル
        max_length (int): 特徴量の最大長さ

    Returns:
        pd.DataFrame: 特徴量とラベルを含むデータフレーム（スペクトログラム、男か女か）
    """
    features = []
    labels = []
    label = 0
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    processed_files = 0
    dir_name = ""


    for root, dirs, files in os.walk(directory):
        
        #ディレクトリ名が性別ラベルにある場合はそのラベルを使用する（音声フォルダの場合にはディレクトリ名が性別ラベルにないため）
        if os.path.basename(os.path.split(root)[0]) in gender_labels:
            dir_name = os.path.basename(os.path.split(root)[0])

        print(dir_name)
        for file_name in files:
            if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                file_path = os.path.join(root, file_name)
                feature = extract_log_mel_spectrogram(file_path, max_length)
                #男性の場合1ラベルを、女性の場合0ラベルを付与する
                label = gender_labels.get(os.path.basename(root), gender_labels.get(dir_name))
                features.append(feature)
                labels.append(label)
            processed_files += 1
            progress_label.config(text=f"データセット作成中: {processed_files}/{total_files}, ディレクトリ名: {dir_name}, ラベル: {label}")
            progress_label.update_idletasks()

    
    return pd.DataFrame({'feature': features, 'label': labels})

def create_model(input_shape: tuple) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
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

def save_dataset(X: np.ndarray, y: np.ndarray, save_path: str):
    """
    データセットを保存する関数
    """
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(save_path, index=False)

def load_dataset(save_path: str):
    df = pd.read_csv(save_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def delete_dataset(save_path: str, delete_label: tk.Label):
    os.remove(save_path)
    delete_label.config(text="データセットを削除しました")
    delete_label.update_idletasks()

def make_model(data_path: str, model_path: str, progress_label: tk.Label, max_length: int, dataset_path: str):
    """
    モデルを作成する関数

    Args:
        data_path (str): 学習データのZIPファイルのパス
        model_path (str): 保存するモデルファイルのパス
        progress_label (tk.Label): 進捗状況を表示するラベル
        max_length (int): 特徴量の最大長さ
        dataset_path (str): データセットの保存パス
    """

    if os.path.exists(dataset_path):
        X, y = load_dataset(dataset_path)
        print(X)
        print(y)
    else:

        # データの解凍
        progress_label.config(text="データを解凍中...")
        progress_label.update_idletasks()

        #もしZIPファイルがある場合は展開する
        if os.path.exists(data_path):
            file_name = os.path.basename(data_path)
            file_name = os.path.splitext(file_name)[0]
        else:
            download_data("https://drive.usercontent.google.com/download?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt&export=download&authuser=0", data_path)

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
        print(f"================================================================")
        print(f"Loaded {len(gender_labels)} gender labels")
        print(f"================================================================")

        progress_label.config(text="データセットを作成中...")
        progress_label.update_idletasks()

        dataset = create_dataset(extracted_file, gender_labels, progress_label, max_length)
        
        # デバッグ情報を追加
        progress_label.config(text=f"データセットのサンプル数: {len(dataset)}")
        progress_label.update_idletasks()
        print(f"================================================================")
        print(f"データセットのサンプル数: {len(dataset)}")
        print(f"================================================================")
        print(dataset.head())   
        print(f"================================================================")

        # データセットが空でないことを確認
        if dataset.empty:
            progress_label.config(text="データセットが空です")
            return

        X = np.array(dataset['feature'].tolist())
        y = dataset['label'].values

        print(X)
        print(y)

        # Xが2次元配列であることを確認
        if X.ndim != 2:
            progress_label.config(text="2次元配列ではありません")
            return

        progress_label.config(text="データセットを保存中...")
        progress_label.update_idletasks()

        #データセットを保存
        save_dataset(X, y, dataset_path)


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
    model = create_model(X_train.shape[1:])

    progress_label.config(text="モデルのトレーニング中...")
    progress_label.update_idletasks()

    # モデルのトレーニング
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    progress_label.config(text="モデルの保存中...")
    progress_label.update_idletasks()

    # モデルの保存
    model.save(model_path.replace('.h5', '.keras'))

    progress_label.config(text="モデルのトレーニング完了！")
    progress_label.update_idletasks()
    print("モデルのトレーニング完了！")