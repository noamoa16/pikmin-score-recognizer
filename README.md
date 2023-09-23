# pikmin-score-recognizer

ピクミン2チャレンジモードのリザルト画面から、スコアを取得するAIです。

**ブラウザ上で実行可能です。実行するには[こちらをクリック！](https://noamoa16.github.io/pikmin-score-recognizer/)**

なお、ステージクリア後のリザルト画面にのみ対応しており、ステージ選択画面に表示されるスコアには対応していません。
将来的に、ピクミン2以外にも対応したいと考えています。

スコアの認識には失敗することがあります。特に以下のような画像は失敗する確率が高くなります。
- 直撮り画像
- 斜めになっている画像
- 不鮮明な画像
- 過度にサイズの大きな画像(例：5504×3096)

## 対応画像形式

- JPEG (.jpg, .jpeg, .jfif)
- PNG (.png)
- ビットマップ (.bmp)

## 謝辞

以下の方々に、リザルト画面をAI学習に使用する許可を頂きました。ありがとうございます。

- こばちのうどん氏
- TEL氏
- サクレカマドフマ氏
- zukki氏
- mercysnow氏
- φ氏
- マイコー氏
- エープリル氏

## 開発者向け

### 他のウェブアプリでスコア認識プログラムを使用する場合

このレポジトリをクローンしてください。

`training/data`フォルダは容量が大きいため、モデルを訓練しない場合は削除するか、`.gitignore`に追加して構いません。

[index.html](index.html)と[js/main.js](js/main.js)を書き換えてください。
ディレクトリ構造を変える場合は、[index.html](index.html)内のパス(`script`タグの`src`と、変数`ONNX_PATH`)を適切に書き換える必要があります。

### ローカル上でのスコア認識の実行

Windows 10で動作確認済みです。他のOSでも動作するはずです。

[index.html](index.html)を直接ブラウザで開いた場合、ローカルファイルを読み込む関係で、セキュリティエラーが発生します。
そのため、ローカルサーバーを立てる必要があります。
`http-server`を用いる場合を以下に示します。
```
# インストール
npm install --global http-server
# 実行
cd [このレポジトリのルートディレクトリ]
http-server
```
上記のコマンドを実行したら、ブラウザ([http://127.0.0.1:8080](http://127.0.0.1:8080))を開きます。
ポートは環境によって異なる場合があります。詳しくは`http-server`の出力を参照してください。

### モデルの訓練

以下の環境で動作確認済みです。
- Google Colaboratory (Linux / Python 3.10)
- Windows 10 / Python 3.11

環境構築が不要なGoogle Colaboratoryでの実行をオススメします。

3.10よりも古いバージョンのPythonだと動作しない可能性があります。

#### Google Colaboratory

[こちら](https://colab.research.google.com/drive/1RSR7jDjRSAYEq0UIhDz-T_Pa7QhOO89O?usp=sharing)から実行できます。

#### PC

Windows 10を想定していますが、他のOSでも動作するはずです。

`python`と`pip`は必要に応じて置き換えてください。

#### 必要なモジュールのインストール

```
pip install -r training/requirements.txt
```

#### 訓練の実行

```
python training/train_model.py
```

### スコア抽出プログラムのコンパイル

[ブラウザ版(JavaScript)のスコア抽出プログラム](cpp/number_extractor.cpp)はC++で記述されており、C++プログラムの変更を反映させるには[emscripten](https://emscripten.org/docs/getting_started/downloads.html)をインストールし、以下のコマンドを実行してコンパイルする必要があります。

```
em++ -O3 ./cpp/number_extractor.cpp -o ./cpp/number_extractor.js -DEMSCRIPTEN -s EXPORTED_FUNCTIONS="['_malloc', '_free']" -s EXPORTED_RUNTIME_METHODS="['cwrap', 'getValue']" -sINITIAL_MEMORY=128MB -sSTACK_SIZE=64MB
```

成功すれば[cpp/number_extractor.js](cpp/number_extractor.js)と[cpp/number_extractor.wasm](cpp/number_extractor.wasm)が生成されます。
