# 声で性別を判定したい
![image](https://github.com/user-attachments/assets/b436b135-b4ed-44c8-9721-88b227cde489)

# インストール
![image](https://github.com/user-attachments/assets/df1fca89-78cf-49e0-a37e-8debaf68621a)
https://github.com/p-nasimonan/Gender-determination-by-voice-GDV-/releases/tag/v1.0.0
- zipファイルを解凍
- GDV.exeをダブルクリックして使う
  
# 使い方
- 実行する
- ファイルを選択する
- ![スクリーンショット 2024-08-17 141011](https://github.com/user-attachments/assets/17f5ecdf-8052-4725-a5c0-3db3f3a66c87)

- 結果が出る
- ![image](https://github.com/user-attachments/assets/7b1db8c7-8fa4-4a4f-923a-88be4fb79ab8)

- 消すボタンを押すと、リセットされる。

  # 注意
  機械学習素人が作ったので精度はよくわかりません。女声を判断したいからなるべく男と判断されるようにしたけど、まだまだです。もしかしたら単純にピッチとフォルマントをifで判断するのもいいのかも。


# 学習した音声データ 
https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

# 参考文献
- https://note.nkmk.me/python-tensorflow-keras-basics/
- https://qiita.com/mhrt-tech-biz-blog/items/be436c12d93ed330f99e

## 今後
どうやったらいいんだー精度が上がらない。ニューラルネットワークはまだ誰からも習ってないからわからん。特徴量の取り出し方に問題があるのか？モデルの作り方はこれであってるのか？もしくは画像認識のモデルを使ってどうにかするとか？
