import os
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import io
import torch
import json

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
        self.scaler_X = None
        self.scaler_y = None
        
        # モデルID（データベース保存用）
        self.db_model_id = None
        self.model_uuid = self.config.get("uuid")
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """データの前処理を行う
        
        Args:
            data: 生データ
            
        Returns:
            前処理済みの特徴量とターゲット
        """
        # 特徴量とターゲットを分離
        if self.features is None:
            # デフォルトの特徴量を使用
            features = ['close', 'high', 'low', 'open', 'volume']
        else:
            features = self.features
            
        X = data[features].values
        y = data['close'].values
        
        # スケーリング
        if self.scaler_X is None:
            self.scaler_X = MinMaxScaler()
            X = self.scaler_X.fit_transform(X)
        else:
            X = self.scaler_X.transform(X)
            
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler()
            y = self.scaler_y.fit_transform(y.reshape(-1, 1))
        else:
            y = self.scaler_y.transform(y.reshape(-1, 1))
            
        # シーケンスデータの作成
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length].flatten())
            y_seq.append(y[i + self.sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self):
        """モデルの構築"""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        
        logger.info(f"RandomForestモデルを構築しました: n_estimators={self.n_estimators}")
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        target_col: str = "close",
        save_model: bool = True
    ) -> Dict[str, Any]:
        """モデルの学習
        
        Args:
            df_train: 学習用データ
            df_val: 検証用データ
            target_col: 予測対象の列名
            save_model: 学習後にモデルを保存するかどうか
            
        Returns:
            学習結果の情報
        """
        # 学習開始時間
        training_start = datetime.now()
        
        # データの前処理
        X_train, y_train = self._preprocess_data(df_train)
        
        # 検証データがあれば前処理
        if df_val is not None:
            X_val, y_val = self._preprocess_data(df_val)
        
        # モデルの構築
        if self.model is None:
            self._build_model()
        
        # 学習の実行
        self.model.fit(X_train, y_train.ravel())
        
        # 学習終了時間
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        # 学習結果
        train_metrics = {
            "training_start": training_start,
            "training_end": training_end,
            "training_duration": training_duration
        }
        
        # 検証データがある場合は評価を実施
        if df_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred = val_pred.reshape(-1, 1)
            val_pred = self.scaler_y.inverse_transform(val_pred)
            y_val_orig = self.scaler_y.inverse_transform(y_val)
            
            # 評価指標の計算
            mse = np.mean((y_val_orig - val_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_val_orig - val_pred))
            
            train_metrics.update({
                "val_mse": mse,
                "val_rmse": rmse,
                "val_mae": mae
            })
        
        # データベースにモデルを保存
        if save_model:
            try:
                session = self.db.get_session()
                
                # モデルの保存
                db_model = DbModel(
                    uuid=self.model_uuid or str(uuid.uuid4()),
                    name=f"LSTM_{self.currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    model_type=self.model_type,
                    currency_pair=self.currency_pair,
                    generation=0,
                    hyperparameters=self.hyperparameters,
                    features=self.features
                )
                
                session.add(db_model)
                session.commit()
                
                # モデルIDを保存
                self.db_model_id = db_model.id
                
                # モデルのメタデータを更新
                self._save_model_metadata(self.db_model_id, training_start, training_end)
                train_metrics["model_saved"] = True
                
                logger.info(f"モデルをデータベースに保存しました: ID={db_model.id}")
                
            except Exception as e:
                logger.error(f"モデルの保存に失敗しました: {str(e)}")
                session.rollback()
            finally:
                session.close()
        
        logger.info(f"モデル学習完了: RMSE={train_metrics.get('val_rmse', 'N/A')}")
        
        return train_metrics
    
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
    
    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """モデルの評価
        
        Args:
            df: 評価用データ
            target_col: 予測対象の列名
            save_to_db: 評価結果をデータベースに保存するかどうか
            
        Returns:
            評価結果
        """
        if self.model is None:
            logger.error("モデルがロードされていません")
            raise ValueError("モデルがロードされていません")
        
        # データの前処理
        X, y_true_scaled = self._preprocess_data(df)
        
        # 予測の実行
        y_pred_scaled = self.model.predict(X)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        # スケール逆変換
        y_true = self.scaler_y.inverse_transform(y_true_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # 評価指標の計算
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # 方向予測の正確さ
        y_true_diff = np.diff(y_true.flatten())
        y_pred_diff = np.diff(y_pred.flatten())
        direction_match = (np.sign(y_true_diff) == np.sign(y_pred_diff))
        direction_accuracy = np.mean(direction_match)
        
        # 評価結果
        evaluation = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "timestamp": datetime.now()
        }
        
        logger.info(f"モデル評価: RMSE={rmse:.6f}, 方向予測精度={direction_accuracy:.2%}")
        
        # データベースに保存
        if save_to_db and self.db_model_id:
            self._save_evaluation_results(evaluation, df.index[0], df.index[-1])
        
        return evaluation
    
    def save(self, path: Optional[str] = None) -> str:
        """モデルの保存
        
        Args:
            path: 保存先パス
            
        Returns:
            保存先パス
        """
        if self.model is None:
            logger.error("保存するモデルがありません")
            raise ValueError("保存するモデルがありません")
        
        if path is None:
            os.makedirs(self.app_config.save_models_path, exist_ok=True)
            path = os.path.join(
                self.app_config.save_models_path,
                f"rf_{self.currency_pair}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # モデル本体の保存
        model_path = f"{path}.joblib"
        joblib.dump(self.model, model_path)
        
        # スケーラーの保存
        scaler_path = f"{path}_scalers.joblib"
        joblib.dump({
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y
        }, scaler_path)
        
        # 設定の保存
        config_path = f"{path}_config.joblib"
        config_to_save = {
            "hyperparameters": self.hyperparameters,
            "model_type": self.model_type,
            "currency_pair": self.currency_pair,
            "features": self.features,
            "sequence_length": self.sequence_length
        }
        joblib.dump(config_to_save, config_path)
        
        logger.info(f"モデルを保存しました: {model_path}")
        
        return model_path
    
    def load(self, model_path: str, config_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
        """モデルの読み込み
        
        Args:
            model_path: モデルファイルのパス
            config_path: 設定ファイルのパス
            scaler_path: スケーラーファイルのパス
        """
        # モデル本体の読み込み
        self.model = joblib.load(model_path)
        
        # 設定の読み込み
        if config_path is None:
            config_path = model_path.replace(".joblib", "_config.joblib")
        
        if os.path.exists(config_path):
            loaded_config = joblib.load(config_path)
            self.hyperparameters = loaded_config.get("hyperparameters", self.hyperparameters)
            self.model_type = loaded_config.get("model_type", self.model_type)
            self.currency_pair = loaded_config.get("currency_pair", self.currency_pair)
            self.features = loaded_config.get("features", self.features)
            self.sequence_length = loaded_config.get("sequence_length", self.sequence_length)
        
        # スケーラーの読み込み
        if scaler_path is None:
            scaler_path = model_path.replace(".joblib", "_scalers.joblib")
        
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler_X = scalers.get("scaler_X")
            self.scaler_y = scalers.get("scaler_y")
        
        logger.info(f"モデルを読み込みました: {model_path}")
    
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