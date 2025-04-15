from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    sequence_length: int
    n_layers: int
    units: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    features: list[str]

def get_lstm_config() -> ModelConfig:
    """LSTMモデルの設定を取得します。"""
    return ModelConfig(
        sequence_length=60,
        n_layers=2,
        units=128,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        features=[
            'open', 'high', 'low', 'close',
            'sma_5', 'sma_20', 'sma_60',
            'rsi_14', 'macd', 'bollinger_upper',
            'bollinger_lower', 'volume'
        ]
    )

def get_sentiment_config() -> ModelConfig:
    """センチメント分析モデルの設定を取得します。"""
    return ModelConfig(
        sequence_length=30,
        n_layers=1,
        units=64,
        dropout_rate=0.3,
        learning_rate=0.0005,
        batch_size=16,
        epochs=50,
        features=[
            'sentiment_score', 'news_count',
            'positive_news_ratio', 'negative_news_ratio'
        ]
    ) 