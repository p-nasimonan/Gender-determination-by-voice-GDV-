import sys #システム関連の機能を提供するライブラリ
import tkinter as tk #GUIライブラリ
import os #OS機能を使用するライブラリ。ファイル名の取得を行うために使用。
import json #JSONファイルの読み込みを行うために使用。
from tkinter import * #tkinterモジュール内で定義されているメソッドや変数をまとめてインポート
from tkinter import filedialog #tkinter内のfiledialogを使用するために記述。
import librosa #音声ファイルの読み込みを行うために使用。
import numpy as np 
import pandas as pd 
import analys
from make_model import make_model

data_path = "data/jvs_ver1.zip"
model_path = "gender_classification_model.h5"

def make_window(width: int, height: int, bg_color: str, text_color: str) -> tk.Tk:
    """
    ウィンドウを作成し、Tkinterオブジェクトを取得する関数

    Args:
        width (int): ウィンドウの幅
        height (int): ウィンドウの高さ
        bg_color (str): 背景色
        text_color (str): テキストの色

    Returns:
        tk.Tk: Tkinterオブジェクト
    """
    result = tk.Tk()

    # ウィンドウにタイトルを設定
    result.title("音声性別判定")   

    # ウィンドウのサイズを設定
    result.geometry(f"{width}x{height}")
    result.minsize(width-100, height-300)    # 最小サイズ

    # ウィンドウの背景色を設定
    result.config(bg=bg_color)

    # ウィンドウのアイコンを設定
    result.iconbitmap(r"icon.ico")

    # 進捗状況を表示するラベル
    progress_label = tk.Label(result, text="", bg=bg_color, fg=text_color)
    progress_label.place(x=100, y=80)

    # モデルを作成するボタン
    make_model_button = tk.Button(result, text="モデルを作成する", command=lambda: make_model(data_path, model_path, progress_label), bg="white", fg="black")
    make_model_button.place(x=100, y=40)

    # ファイルパスのボックスとファイルを開くボタンを作成
    box_and_file_open_button(result, "音声ファイルを開く", bg_color, text_color, 100, 100)

    return result


def box_and_file_open_button(window: tk.Tk, text: str, bg_color: str, text_color: str, x: int, y: int):
    file_path_var = tk.StringVar()  # StringVarを使用してファイルパスを管理

    def click_file_open_button():
        try:
            files_path = filedialog.askopenfilenames(filetypes=[("音声ファイル", "*.wav")])
            if files_path == () or files_path is None:
                print("ファイルパスが空です")
                return
        except:
            print("ファイルを開くことができませんでした")
            return
        file_path_var.set(files_path)  # StringVarにファイルパスを設定
        for index, file_path in enumerate(files_path):
            file, gender = analys.what_is_gender(file_path, data_path, model_path) #解析
        
        # ファイル名と性別を表示
        label = tk.Label(window, text=f"{file}: {gender}", bg=bg_color, fg=text_color)
        label.place(x=x-100, y=(y+100)+(20*index))
        label_x, label_y = x-100, (y+100)+(20*index)
        label.place(x=label_x, y=label_y)
    
    #ファイルパスが表示されるボックス
    file_path_box = tk.Entry(window, width=40, textvariable=file_path_var, bg=bg_color, fg=text_color)
    file_path_box.place(x=x, y=y)

    #ファイルを開くボタン
    open_button = tk.Button(window, text=text, command=click_file_open_button, bg="white", fg="black")
    open_button.place(x=x+300, y=y)

    return None

def read_config() -> tuple:
    with open("config.json", "r") as f:
        config = json.load(f)
    theme = config["theme"]
    learning_data = config["learning_data"]
    return theme, learning_data


def start(theme: str):
    # テーマ別にウィンドウを作成
    if theme == "light":
        window = make_window(600, 500, "white", "black")
    elif theme == "dark":
        window = make_window(600, 500, "#222222", "white")
    else:
        print("テーマエラー: config.jsonのthemeを確認してください")
        sys.exit()

    # ウィンドウのループ処理
    window.mainloop()


if __name__ == "__main__":
    theme, analys.learning_data = read_config()
    
    start(theme)