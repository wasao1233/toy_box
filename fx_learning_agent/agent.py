import datetime
import pandas as pd
from typing import Optional
from log import logger
from models.lstm_model import LSTMModel
from model_config import get_lstm_config, get_sentiment_config

class Agent:
    def train_model(self, model_type: str, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> int:
        """
        モデルを学習する
        
        Args:
            model_type: モデルの種類（'lstm'または'sentiment'）
            training_data: 学習データ
            validation_data: 検証データ（Noneの場合は自動的に分割）
            
        Returns:
            int: 学習したモデルのID
        """
        try:
            # モデルの設定を取得
            if model_type == 'lstm':
                model_config = get_lstm_config()
            elif model_type == 'sentiment':
                model_config = get_sentiment_config()
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
            
            # モデルのインスタンスを作成
            model = LSTMModel(model_config)
            
            # 学習開始時間を記録
            training_start = datetime.datetime.now()
            
            # モデルを学習
            model.train(training_data, validation_data)
            
            # 学習終了時間を記録
            training_end = datetime.datetime.now()
            
            # モデルをデータベースに保存
            model_id = model._save_model_metadata(training_start, training_end)
            
            return model_id
            
        except Exception as e:
            logger.error(f"モデルの学習に失敗しました: {str(e)}")
            raise 