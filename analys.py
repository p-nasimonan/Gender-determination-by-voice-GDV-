"""
トレーニング済みモデルを使用して音声ファイルの性別を分類するスクリプト
"""

from make_model import extract_log_mel_spectrogram

import tensorflow as tf
import librosa
import soundfile as sf
import os

def load_model(model_path: str) -> tf.keras.Model:
    """
    トレーニング済みモデルを読み込む関数

    Args:
        model_path (str): モデルファイルのパス

    Returns:
        tf.keras.Model: 読み込まれたモデル
    """
    return tf.keras.models.load_model(model_path)

def cut_audio(file_path: str) -> str:
    """
    音声ファイルから声を切り出し、temp_wavとして返す関数
    """
    # 音声ファイルを読み込む
    y, sr = librosa.load(file_path)
    #無音の部分を消す
    y, index = librosa.effects.trim(y, top_db=20)

    # 音声ファイルの長さを取得
    duration = librosa.get_duration(y=y, sr=sr)
    # 音声ファイルの長さが30秒以上の場合は、30秒まで切り出す
    if duration > 30:
        y = y[:30*sr]

    # 音声ファイルを保存
    temp_wav_path = "tmp/"+os.path.basename(file_path)
    sf.write(temp_wav_path, y, sr)
    return temp_wav_path

def classify_gender(file_path: str, model: tf.keras.Model, max_length: int) -> str:
    """
    音声ファイルの性別を分類する関数

    Args:
        file_path (str): 音声ファイルのパス
        model (tf.keras.Model): トレーニング済みモデル
        max_length (int): 特徴量の最大長

    Returns:
        prediction float: 性別の確率
    """
    temp_wav = cut_audio(file_path)
    feature = extract_log_mel_spectrogram(temp_wav, max_length)
    feature = feature.reshape(1, -1)  # モデルの入力形状に合わせてリシェイプ
    prediction = model.predict(feature)
    return prediction

def what_is_gender(file_path: str, model_path: str, max_length: int) -> str:
    """
    音声ファイルの性別を分類する関数
    Args:
        file_path (str): 音声ファイルのパス
        model_path (str): モデルファイルのパス
        max_length (int): 特徴量の最大長

    Returns:
        str: 性別 ('男' または '女')
    """
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("モデルが存在しません。")
        return None

    # 性別分類の実行
    gender_prediction = classify_gender(file_path, model, max_length)
    if gender_prediction[0][0] > 0.2:
        gender = "男"
    else:
        gender = "女"
    return gender