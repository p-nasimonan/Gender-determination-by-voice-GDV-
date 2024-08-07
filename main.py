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
# import openai
# openai.api_key = "あなたのOpenAIAPIキー"

def make_window(width: int, height: int, bg_color: str, text_color: str) -> tk.Tk:
    # ウィンドウの作成と、Tkinterオブジェクトの取得
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

    audio, sr = box_and_file_open_button(result, "ファイルを開く", bg_color, text_color, 100, 100)




    return result


def output_txtbox(txt_box: tk.Entry, text: str):
    txt_box.delete(0, tk.END)
    txt_box.insert(0, text)

def click_file_open_button():
    file_path = filedialog.askopenfilename(filetypes=[("音声ファイル", "*.wav;*.mp3")])
    return file_path




def box_and_file_open_button(window: tk.Tk, text: str, bg_color: str, text_color: str, x: int, y: int) -> list:
    file_path_var = tk.StringVar()  # StringVarを使用してファイルパスを管理

    file_path_box = tk.Entry(window, width=50, textvariable=file_path_var, bg=bg_color, fg=text_color)
    file_path_box.place(x=x, y=y)

    def click_file_open_button():
        files_path = filedialog.askopenfilenames(filetypes=[("音声ファイル", "*.wav;*.mp3")])
        file_path_var.set(files_path)  # StringVarにファイルパスを設定
        for index, file_path in enumerate(files_path):
            file, gender = analys.what_is_gender(file_path) #解析
            
            # ファイル名と性別を表示
            label = tk.Label(window, text=f"{file}: {gender}", bg=bg_color, fg=text_color)
            label.place(x=x-100, y=(y+100)+(20*index))
            label_x, label_y = x-100, (y+100)+(20*index)
            label.place(x=label_x, y=label_y)

    open_button = tk.Button(window, text=text, command=click_file_open_button, bg=bg_color, fg=text_color)
    open_button.place(x=x+300, y=y)

    return file_path_box, open_button


def read_config() -> str:
    with open("config.json", "r") as f:
        config = json.load(f)
    theme = config["theme"]
    model = config["model"]
    learning_data = config["learning_data"]
    return theme, model, learning_data


def start(theme: str):
    # テーマ別にウィンドウを作成
    if theme == "light":
        window = make_window(500, 500, "white", "black")
    elif theme == "dark":
        window = make_window(500, 500, "#222222", "white")

    # ウィンドウのループ処理
    window.mainloop()


if __name__ == "__main__":
    theme, analys.model, analys.learning_data = read_config()
    
    start(theme)
