"""
トレーニング済みモデルを使用して音声ファイルの性別を分類するスクリプト
"""

from make_model import extract_mel_spectrogram, make_model

import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

def load_model(model_path: str) -> tf.keras.Model:
    """
    トレーニング済みモデルを読み込む関数

    Args:
        model_path (str): モデルファイルのパス

    Returns:
        tf.keras.Model: 読み込まれたモデル
    """
    return tf.keras.models.load_model(model_path)

def classify_gender(file_path: str, model: tf.keras.Model, scaler: StandardScaler) -> str:
    """
    音声ファイルの性別を分類する関数

    Args:
        file_path (str): 音声ファイルのパス
        model (tf.keras.Model): トレーニング済みモデル
        scaler (StandardScaler): データの標準化器

    Returns:
        str: 性別 ('male' または 'female')
    """
    feature = extract_features(file_path)
    feature = scaler.transform([feature])
    prediction = model.predict(feature)
    return 'male' if prediction[0] > 0.5 else 'female'

def what_is_gender(file_path: str, data_path: str, model_path: str) -> str:
    """
    音声ファイルの性別を分類する関数
    Args:
        file_path (str): 音声ファイルのパス
        data_path (str): 学習データのパス
        model_path (str): モデルファイルのパス
    Returns:
        str: 性別 ('male' または 'female')
    """
    # モデルと標準化器の読み込み
    try:
        model = load_model(model_path)
    except:
        print("モデルの読み込みに失敗しました")
        return make_model(data_path, model_path)
    try:
        scaler = StandardScaler()
        scaler.fit(np.array([np.fromstring(feature[1:-1], sep=' ') for feature in pd.read_csv('learning_data.csv')['feature']]))
    except:
        print("標準化器の読み込みに失敗しました")
        return None

    # 性別分類の実行
    gender = classify_gender(file_path, model, scaler)
    return gender