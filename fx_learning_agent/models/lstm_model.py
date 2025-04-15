import os
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import io
import torch
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.logger import get_logger
from config.config import get_config
from models.data_models import Model as DbModel, ModelPerformance, Prediction
from utils.database import get_db

logger = get_logger(__name__)

class TimeSeriesModel:
    """時系列予測モデル
    
    RandomForestを使用して時系列データの予測を行います。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化
        
        Args:
            config: モデル設定
        """
        self.app_config = get_config()
        self.db = get_db()
        
        # モデル設定
        self.config = config or {}
        self.model_type = self.config.get("model_type", "RandomForest")
        self.currency_pair = self.config.get("currency_pair", "USD/JPY")
        
        # ハイパーパラメータ
        self.hyperparameters = self.config.get("hyperparameters", {})
        
        # デフォルトのハイパーパラメータ
        self.sequence_length = self.hyperparameters.get("sequence_length", 10)
        self.n_estimators = self.hyperparameters.get("n_estimators", 100)
        self.max_depth = self.hyperparameters.get("max_depth", None)
        self.min_samples_split = self.hyperparameters.get("min_samples_split", 2)
        self.min_samples_leaf = self.hyperparameters.get("min_samples_leaf", 1)
        
        # 特徴量
        self.features = self.config.get("features", None)
        
        # モデルオブジェクト
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler = None
        
        # モデルID（データベース保存用）
        self.db_model_id = None
        self.model_uuid = self.config.get("uuid")
        
        # 市場指数データのスケーラー
        self.market_scalers = {}
        
        # 方向性の精度を追加
        self.direction_accuracy = None

    def _preprocess_data(self, data: pd.DataFrame, market_indices: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """データの前処理を行います。

        Args:
            data (pd.DataFrame): 前処理を行うデータフレーム
            market_indices (Optional[Dict[str, pd.DataFrame]], optional): 市場指数データ

        Returns:
            Tuple[np.ndarray, np.ndarray]: 前処理済みの特徴量とターゲット
        """
        # 基本特徴量の準備
        features = data[['open', 'high', 'low', 'close', 'volume']].values

        # 市場指数データの追加（存在する場合）
        if market_indices is not None:
            for index_name, index_data in market_indices.items():
                merged_data = pd.merge(data, index_data, left_index=True, right_index=True, how='left')
                merged_data = merged_data[f'close_{index_name}'].fillna(method='ffill').fillna(method='bfill')
                features = np.column_stack((features, merged_data.values))

        # スケーリング
        if not hasattr(self, 'scaler_X') or self.scaler_X is None:
            self.scaler_X = StandardScaler()
            features_scaled = self.scaler_X.fit_transform(features)
        else:
            features_scaled = self.scaler_X.transform(features)

        # ターゲット値の準備
        target = data['close'].values.reshape(-1, 1)
        if not hasattr(self, 'scaler_y') or self.scaler_y is None:
            self.scaler_y = StandardScaler()
            target_scaled = self.scaler_y.fit_transform(target)
        else:
            target_scaled = self.scaler_y.transform(target)

        # シーケンスデータの作成
        X, y = [], []
        # 特徴量とターゲットを準備
        features = []
        target = data['close'].values

        # 基本的な特徴量を追加
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                features.append(data[col].values)

        # 市場指数データを追加
        if market_indices is not None:
            for index_name, index_data in market_indices.items():
                if 'close' in index_data.columns:
                    # インデックスデータを日付でマージ
                    merged_data = pd.merge(data, index_data[['close']], 
                                         left_index=True, right_index=True, 
                                         how='left', suffixes=('', f'_{index_name}'))
                    # NaNを前方補完と後方補完で埋める
                    merged_data = merged_data[f'close_{index_name}'].fillna(method='ffill').fillna(method='bfill')
                    features.append(merged_data.values)

        # 特徴量を結合
        X = np.column_stack(features)

        # スケーリング
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # シーケンスデータの作成
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(target[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # シーケンスデータを2次元に変換
        n_samples = X_seq.shape[0]
        X_seq = X_seq.reshape(n_samples, -1)

        return X_seq, y_seq
    
    def _build_model(self, input_shape: Tuple[int, ...]) -> RandomForestRegressor:
        """モデルの構築"""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        
        logger.info(f"RandomForestモデルを構築しました: n_estimators={self.n_estimators}")
        
        return model
    
    def fit(
        self,
        data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]] = None,
        market_indices: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """モデルの学習を実行
        
        Args:
            data: 学習データ（DataFrameまたは(X, y)のタプル）
            validation_data: 検証データ（DataFrameまたは(X, y)のタプル）
            market_indices: 市場指数データ
            **kwargs: その他の引数
            
        Returns:
            学習結果
        """
        try:
            # データの前処理
            if isinstance(data, tuple):
                X_processed, y_processed = data
            else:
                X_processed, y_processed = self._preprocess_data(data=data, market_indices=market_indices)
            
            if validation_data is not None:
                if isinstance(validation_data, tuple):
                    X_val_processed, y_val_processed = validation_data
                else:
                    X_val_processed, y_val_processed = self._preprocess_data(
                        data=validation_data,
                        market_indices=market_indices
                    )
                validation_data = (X_val_processed, y_val_processed)
            
            # モデルの構築
            self.model = self._build_model(X_processed.shape[1:])
            
            # 学習の実行
            self.model.fit(X_processed, y_processed)
            
            # モデルの評価
            if validation_data is not None:
                val_mse, val_rmse, val_mae = self.evaluate(
                    validation_data,
                    market_indices=market_indices
                )
            else:
                val_mse, val_rmse, val_mae = self.evaluate(
                    data,
                    market_indices=market_indices
                )
            
            # モデルの保存
            self.save()
            
            # 学習結果の記録
            result = {
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'history': {}  # RandomForestRegressorはhistoryを持たない
            }
            
            return result
            
        except Exception as e:
            logger.error(f"モデル学習中にエラーが発生しました: {str(e)}")
            raise
    
    def predict(
        self,
        df: pd.DataFrame,
        target_col: str = "close"
    ) -> Tuple[np.ndarray, List[datetime]]:
        """予測の実行
        
        Args:
            df: 予測用データ
            target_col: 予測対象の列名
            
        Returns:
            predictions: 予測値の配列
            timestamps: タイムスタンプのリスト
        """
        if self.model is None:
            logger.error("モデルがロードされていません")
            raise ValueError("モデルがロードされていません")
        
        # データの前処理
        X, _ = self._preprocess_data(df)
        
        # 予測の実行
        y_pred_scaled = self.model.predict(X)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        # スケール逆変換
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred, df.index.tolist()
    
    def evaluate(self, data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]], market_indices: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
        """モデルを評価し、MSE、RMSE、MAEを計算します。

        Args:
            data: 評価データ（DataFrameまたは(X, y)のタプル）
            market_indices: 市場指数データ（オプション）

        Returns:
            Tuple[float, float, float]: MSE、RMSE、MAE
        """
        try:
            if isinstance(data, tuple):
                X_processed, y_processed = data
            else:
                X_processed, y_processed = self._preprocess_data(data, market_indices)

            # 予測を実行
            y_pred = self.model.predict(X_processed)

            # スケーリングを元に戻す
            # 予測値を元のスケールに戻すために、同じ次元のゼロ行列を作成
            y_pred_full = np.zeros((y_pred.shape[0], self.scaler.scale_.shape[0]))
            # 予測値を適切な位置（close価格の位置）に配置
            y_pred_full[:, 3] = y_pred
            # スケーリングを元に戻す
            y_pred = self.scaler.inverse_transform(y_pred_full)[:, 3]
            y_true = y_processed

            # 評価指標を計算
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            return mse, rmse, mae

        except Exception as e:
            logger.error(f"モデル評価中にエラーが発生しました: {str(e)}")
            raise
    
    def save(self) -> int:
        """モデルをデータベースに保存
        
        Returns:
            保存されたモデルのID
        """
        if self.model is None:
            logger.error("保存するモデルがありません")
            raise ValueError("保存するモデルがありません")
        
        try:
            session = self.db.get_session()
            
            # モデルとスケーラーをバイナリデータに変換
            model_bytes = io.BytesIO()
            joblib.dump(self.model, model_bytes)
            model_bytes.seek(0)
            
            scaler_bytes = io.BytesIO()
            joblib.dump({
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "market_scalers": self.market_scalers
            }, scaler_bytes)
            scaler_bytes.seek(0)
            
            # モデルメタデータの作成
            model = DbModel(
                name=f"rf_{self.currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type="RandomForest",
                currency_pair=self.currency_pair,
                model_data=model_bytes.getvalue(),
                scaler_data=scaler_bytes.getvalue(),
                hyperparameters=json.dumps(self.hyperparameters),
                features=json.dumps(self.features) if self.features else None,
                sequence_length=self.sequence_length,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            session.add(model)
            session.commit()
            
            self.db_model_id = model.id
            logger.info(f"モデルをデータベースに保存しました: ID={model.id}")
            
            return model.id
            
        except Exception as e:
            logger.error(f"モデルの保存に失敗しました: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def load(self, model_id: int, config_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
        """モデルの読み込み
        
        Args:
            model_id: モデルID
            config_path: 設定ファイルのパス
            scaler_path: スケーラーファイルのパス
        """
        try:
            session = self.db.get_session()
            model = session.query(DbModel).filter_by(id=model_id).first()
            if model and model.model_data and model.scaler_data:
                # モデルを読み込む
                model_bytes = io.BytesIO(model.model_data)
                self.model = joblib.load(model_bytes)
                
                # スケーラーを読み込む
                scaler_bytes = io.BytesIO(model.scaler_data)
                scalers = joblib.load(scaler_bytes)
                self.scaler_X = scalers.get("scaler_X")
                self.scaler_y = scalers.get("scaler_y")
                self.market_scalers = scalers.get("market_scalers", {})
                
                # 設定の読み込み
                if config_path is None:
                    config_path = f"{model.name}_config.joblib"
                
                if os.path.exists(config_path):
                    loaded_config = joblib.load(config_path)
                    self.hyperparameters = loaded_config.get("hyperparameters", self.hyperparameters)
                    self.model_type = loaded_config.get("model_type", self.model_type)
                    self.currency_pair = loaded_config.get("currency_pair", self.currency_pair)
                    self.features = loaded_config.get("features", self.features)
                    self.sequence_length = loaded_config.get("sequence_length", self.sequence_length)
                
                logger.info(f"モデルを読み込みました: ID={model_id}")
            else:
                logger.error(f"モデルデータが見つかりません: ID={model_id}")
        except Exception as e:
            logger.error(f"モデルの読み込みに失敗しました: {str(e)}")
            raise
        finally:
            session.close()
    
    def _save_model_metadata(self, model_id: int, training_start: datetime, training_end: datetime) -> None:
        """モデルのメタデータをデータベースに保存"""
        try:
            # モデルとスケーラーをバイナリデータに変換
            model_bytes = io.BytesIO()
            joblib.dump(self.model, model_bytes)
            model_bytes.seek(0)
            
            scaler_bytes = io.BytesIO()
            joblib.dump(self.scaler_X, scaler_bytes)
            scaler_bytes.seek(0)
            
            # モデルメタデータを更新
            model = self.db.get_session().query(DbModel).filter_by(id=model_id).first()
            if model:
                model.model_data = model_bytes.getvalue()
                model.scaler_data = scaler_bytes.getvalue()
                model.training_start = training_start
                model.training_end = training_end
                model.training_duration = (training_end - training_start).total_seconds()
                model.updated_at = datetime.now()
                self.db.get_session().commit()
                logger.info(f"モデルメタデータを更新しました: {model.name}")
            else:
                logger.error(f"モデルが見つかりません: ID={model_id}")
        except Exception as e:
            logger.error(f"モデルメタデータの保存に失敗しました: {str(e)}")
            self.db.get_session().rollback()
            raise

    def load_model(self, model_id: int) -> None:
        """モデルをデータベースから読み込む"""
        try:
            model = self.db.get_session().query(DbModel).filter_by(id=model_id).first()
            if model and model.model_data and model.scaler_data:
                # モデルを読み込む
                model_bytes = io.BytesIO(model.model_data)
                self.model = joblib.load(model_bytes)
                
                # スケーラーを読み込む
                scaler_bytes = io.BytesIO(model.scaler_data)
                self.scaler_X = joblib.load(scaler_bytes)
                
                logger.info(f"モデルを読み込みました: {model.name}")
            else:
                logger.error(f"モデルデータが見つかりません: ID={model_id}")
        except Exception as e:
            logger.error(f"モデルの読み込みに失敗しました: {str(e)}")
            raise
    
    def _save_evaluation_results(
        self,
        evaluation: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """評価結果をデータベースに保存
        
        Args:
            evaluation: 評価結果
            start_date: 評価期間の開始日
            end_date: 評価期間の終了日
        """
        try:
            session = self.db.get_session()
            
            # 評価結果の保存
            performance = ModelPerformance(
                model_id=self.db_model_id,
                evaluation_type="validation",
                start_date=start_date,
                end_date=end_date,
                profit=float(evaluation.get('profit', 0.0)),
                profit_percent=float(evaluation.get('profit_percent', 0.0)),
                win_count=int(evaluation.get('win_count', 0)),
                loss_count=int(evaluation.get('loss_count', 0)),
                win_rate=float(evaluation.get('win_rate', 0.0)),
                avg_win=float(evaluation.get('avg_win', 0.0)) if evaluation.get('avg_win') is not None else None,
                avg_loss=float(evaluation.get('avg_loss', 0.0)) if evaluation.get('avg_loss') is not None else None,
                max_drawdown=float(evaluation.get('max_drawdown', 0.0)) if evaluation.get('max_drawdown') is not None else None,
                sharpe_ratio=float(evaluation.get('sharpe_ratio', 0.0)) if evaluation.get('sharpe_ratio') is not None else None,
                sortino_ratio=float(evaluation.get('sortino_ratio', 0.0)) if evaluation.get('sortino_ratio') is not None else None,
                accuracy=float(evaluation.get('direction_accuracy', 0.0)) if evaluation.get('direction_accuracy') is not None else None,
                precision=float(evaluation.get('precision', 0.0)) if evaluation.get('precision') is not None else None,
                recall=float(evaluation.get('recall', 0.0)) if evaluation.get('recall') is not None else None,
                f1_score=float(evaluation.get('f1_score', 0.0)) if evaluation.get('f1_score') is not None else None,
                fitness_score=float(evaluation.get('fitness_score', 0.0)),
                details=json.dumps({
                    'mse': float(evaluation.get('mse', 0.0)),
                    'rmse': float(evaluation.get('rmse', 0.0)),
                    'mae': float(evaluation.get('mae', 0.0))
                })
            )
            
            session.add(performance)
            session.commit()
            
            logger.info(f"モデル評価結果をデータベースに保存しました: モデルID={self.db_model_id}")
            
        except Exception as e:
            logger.error(f"評価結果の保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close()
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        timestamps: List[datetime]
    ) -> None:
        """予測結果をデータベースに保存
        
        Args:
            predictions: 予測値
            actual_values: 実際の値
            timestamps: タイムスタンプ
        """
        if self.db_model_id is None:
            logger.warning("モデルIDがないため予測結果を保存できません")
            return
        
        try:
            session = self.db.get_session()
            prediction_time = datetime.now()
            
            for i, (pred, actual, ts) in enumerate(zip(predictions, actual_values, timestamps)):
                # 予測方向（上昇/下降）
                if i > 0:
                    prev_actual = actual_values[i-1][0]
                    actual_change = actual[0] - prev_actual
                    pred_change = pred[0] - prev_actual
                    
                    actual_direction = "up" if actual_change > 0 else "down"
                    predicted_direction = "up" if pred_change > 0 else "down"
                    is_correct = actual_direction == predicted_direction
                    
                    # 変化率（％）
                    actual_change_pct = actual_change / prev_actual * 100
                    pred_change_pct = pred_change / prev_actual * 100
                    
                    # 予測誤差
                    error = abs((pred[0] - actual[0]) / actual[0] * 100)  # パーセント誤差
                else:
                    # 初回データは方向判定できないため中立
                    actual_direction = "neutral"
                    predicted_direction = "neutral"
                    is_correct = None
                    actual_change_pct = 0
                    pred_change_pct = 0
                    error = 0
                
                # 予測結果の保存
                prediction = Prediction(
                    model_id=self.db_model_id,
                    timestamp=prediction_time,
                    target_timestamp=ts,
                    symbol=self.currency_pair,
                    
                    # 予測値
                    predicted_direction=predicted_direction,
                    predicted_change=pred_change_pct,
                    predicted_price=float(pred[0]),
                    confidence=0.5,  # RandomForestではデフォルトで0.5とする
                    
                    # 実際の値
                    actual_direction=actual_direction,
                    actual_change=actual_change_pct,
                    actual_price=float(actual[0]),
                    
                    # 結果評価
                    is_correct=is_correct,
                    error=error
                )
                
                session.add(prediction)
            
            session.commit()
            logger.info(f"{len(predictions)}件の予測結果をデータベースに保存しました")
            
        except Exception as e:
            logger.error(f"予測結果の保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close() 