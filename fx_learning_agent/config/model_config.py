from typing import Dict, Any
from dataclasses import dataclass, field
import uuid

@dataclass
class ModelConfig:
    """モデル設定クラス"""
    model_type: str  # "LSTM" または "SENTIMENT"
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # LSTMモデル用パラメータ
    sequence_length: int = 60  # 時系列の長さ
    hidden_size: int = 64  # 隠れ層のユニット数
    num_layers: int = 2  # LSTMレイヤーの数
    dropout: float = 0.2  # ドロップアウト率
    learning_rate: float = 0.001  # 学習率
    batch_size: int = 32  # バッチサイズ
    epochs: int = 100  # エポック数
    early_stopping_patience: int = 10  # 早期終了の待機エポック数
    
    # センチメントモデル用パラメータ
    model_name: str = "distilbert-base-uncased"  # 事前学習済みモデル名
    max_length: int = 512  # 最大トークン長
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            "model_type": self.model_type,
            "uuid": self.uuid,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "model_name": self.model_name,
            "max_length": self.max_length
        }

def get_lstm_config() -> ModelConfig:
    """LSTMモデルのデフォルト設定を取得"""
    return ModelConfig(model_type="LSTM")

def get_sentiment_config() -> ModelConfig:
    """センチメントモデルのデフォルト設定を取得"""
    return ModelConfig(model_type="SENTIMENT") 