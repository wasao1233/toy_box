# FX Learning Agent

為替市場のデータとニュースを解析し、取引戦略を学習するエージェントです。過去のデータを利用した取引デモ、そして進化的手法を活用して世代ごとに成績を向上させることを目指します。

## 機能

- **データ取得**：Alpha Vantage APIとNews APIを利用してデータを取得
- **学習モデル**：LSTMを使用した時系列予測と、BERTを使用したニュース感情分析
- **進化的学習**：遺伝的アルゴリズムによるモデルの進化
- **バックテスト**：過去データを使用した取引シミュレーション

## 必要条件

- Python 3.8以上
- PostgreSQL（または必要に応じてSQLite）
- 各種APIキー：
  - Alpha Vantage API
  - News API

## インストール

1. リポジトリをクローン

```bash
git clone <repository-url>
cd fx_learning_agent
```

2. 仮想環境を作成して有効化

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

4. 環境変数を設定

`.env`ファイルを作成し、以下の内容を設定します。

```ini
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
NEWS_API_KEY=your_news_api_key

# Database Settings
DB_ENGINE=postgresql # or sqlite
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=postgres
DB_NAME=fx_learning
```

## 使い方

### データの取得

為替レートデータを取得：

```bash
python main.py fetch --pair USD/JPY --timeframe 1h --days 30
```

ニュースデータを取得：

```bash
python main.py news --keywords USD/JPY FRB inflation --days 7
```

### モデルの学習

単一モデルの学習：

```bash
python main.py train --pair USD/JPY --model LSTM
```

### 進化的学習

進化的アルゴリズムを使用したモデルの学習：

```bash
python main.py evolve --pair USD/JPY --model LSTM --generations 10 --population 20
```

### バックテスト

学習したモデルのバックテスト：

```bash
python main.py backtest --model-id 1
```

## プロジェクト構成

```
fx_learning_agent/
├── config/         # 設定ファイル
├── data/           # データ取得・処理関連モジュール
├── models/         # 学習モデル関連モジュール
├── utils/          # ユーティリティモジュール
├── main.py         # メインスクリプト
├── requirements.txt # 依存パッケージ
└── README.md       # 本ファイル
```

## ライセンス

[MIT](LICENSE)

## 謝辞

- Alpha Vantage API
- News API
- その他、利用しているオープンソースライブラリ 