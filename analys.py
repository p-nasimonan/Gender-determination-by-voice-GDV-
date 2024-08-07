"""
今のところ機械学習が全く分からず死んだ

"""


import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from preprocess import preprocess_audio_files
from model import create_model





learning_data, model = "", ""

def remove_silence(input_path: str, output_path: str) -> str:
    # 音声ファイルを読み込み
    sound = AudioSegment.from_file(input_path)

    # 無音部分を検出し、音声を分割
    chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-55, keep_silence=100)

    # 無音部分を除去した新しい音声を作成
    no_silence_audio = AudioSegment.empty()
    for chunk in chunks:
        no_silence_audio += chunk

    # 無音部分を除去した音声を出力
    no_silence_audio.export(output_path, format="mp3")

    org_ms = len(no_silence_audio)
    print('removed: {:.2f} [min]'.format(org_ms/60/1000))
    return output_path


def read_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    if file_path.endswith(".wav") or file_path.endswith(".mp3"):
        tmp_file = remove_silence(file_path, "tmp/no_silence_audio.mp3")
        audio, sr = librosa.load(tmp_file)
    else:
        print("音声ファイルではありません")
    
    return audio, sr

def extract_log_mel_spectrogram(file_path: str) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def preprocess_audio_files(file_paths: list) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for file_path in file_paths:
        log_mel_spectrogram = extract_log_mel_spectrogram(file_path)
        X.append(log_mel_spectrogram)
        # 性別ラベルをファイル名から取得（例: "male_001.wav" -> "male"）
        label = os.path.basename(file_path).split('_')[0]
        y.append(0 if label == 'male' else 1)
    return np.array(X), np.array(y)


def create_model(input_shape: tuple) -> tf.keras.Model:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def what_is_gender(file_path: str) -> str:
    # メルスペクトラムを計算
    audio, sr = read_audio_file(file_path)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 時間平均を取る
    log_mel_spectrogram = np.mean(log_mel_spectrogram, axis=1)
    
    # フラット化してリストに変換
    log_mel_spectrogram = log_mel_spectrogram.flatten().tolist()
    

    file = file_path.split("/")[-1]
    
    # どうにか性別を判定したい
    gender = "male"



    return file, gender
