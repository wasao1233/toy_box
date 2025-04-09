import os
import numpy as np
import pandas as pd
import uuid

# TensorFlowをモックに置き換え
class MockTensorFlow:
    class keras:
        class models:
            Sequential = object
            load_model = lambda x: None
        
        class layers:
            LSTM = object
            Dense = object
            Dropout = object
            BatchNormalization = object
            
        class callbacks:
            EarlyStopping = object
            ModelCheckpoint = object
            ReduceLROnPlateau = object
            
        class optimizers:
            Adam = lambda learning_rate=0.001: None

# モックオブジェクトをtfとして使用
tf = MockTensorFlow()

import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import MinMaxScaler
import joblib

from utils.logger import get_logger
from config.config import get_config
from models.data_models import Model as DbModel, ModelPerformance, Prediction
from utils.database import get_db

logger = get_logger(__name__)

class LSTMModel:
    """LSTM（Long Short-Term Memory）モデル
    
    時系列データ予測のためのディープラーニングモデルを実装します。
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
        self.model_type = self.config.get("model_type", "LSTM")
        self.currency_pair = self.config.get("currency_pair", "USD/JPY")
        
        # ハイパーパラメータ
        self.hyperparameters = self.config.get("hyperparameters", {})
        
        # デフォルトのハイパーパラメータ
        self.sequence_length = self.hyperparameters.get("sequence_length", 60)  # 系列長
        self.n_layers = self.hyperparameters.get("n_layers", 2)  # LSTMレイヤー数
        self.units = self.hyperparameters.get("units", 64)  # LSTM隠れ層のユニット数
        self.dropout_rate = self.hyperparameters.get("dropout_rate", 0.2)  # ドロップアウト率
        self.learning_rate = self.hyperparameters.get("learning_rate", 0.001)  # 学習率
        self.batch_size = self.hyperparameters.get("batch_size", 32)  # バッチサイズ
        self.epochs = self.hyperparameters.get("epochs", 100)  # エポック数
        
        # 特徴量
        self.features = self.config.get("features", None)
        
        # モデルオブジェクト
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
        # モデルID（データベース保存用）
        self.db_model_id = None
        self.model_uuid = self.config.get("uuid")
    
    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str = "close"
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime.datetime]]:
        """データの前処理
        
        Args:
            df: 為替レートのデータフレーム
            target_col: 予測対象の列名
            
        Returns:
            X: 入力データ（3次元配列: [サンプル数, 時間ステップ, 特徴量数]）
            y: 正解データ（2次元配列: [サンプル数, 1]）
            timestamps: タイムスタンプのリスト
        """
        # 前処理用のデータフレーム
        df_copy = df.copy()
        
        # 特徴量のリスト
        if self.features is None:
            # デフォルトでOHLCを使用
            feature_cols = ["open", "high", "low", "close"]
        else:
            feature_cols = self.features
            
        # 利用可能な特徴量だけを使用
        available_features = [col for col in feature_cols if col in df_copy.columns]
        if not available_features:
            logger.error("有効な特徴量がありません")
            raise ValueError("有効な特徴量がありません")
        
        # テクニカル指標の計算
        self._add_technical_indicators(df_copy)
        
        # 欠損値の処理
        df_copy = df_copy.dropna()
        
        # スケーリング
        if self.scaler_X is None:
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            self.scaler_X.fit(df_copy[available_features])
        
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            self.scaler_y.fit(df_copy[[target_col]])
        
        X_scaled = self.scaler_X.transform(df_copy[available_features])
        y_scaled = self.scaler_y.transform(df_copy[[target_col]])
        
        # 時系列データ変換
        X, y, timestamps = [], [], []
        for i in range(len(df_copy) - self.sequence_length):
            X.append(X_scaled[i:i+self.sequence_length])
            y.append(y_scaled[i+self.sequence_length])
            timestamps.append(df_copy.index[i+self.sequence_length])
        
        return np.array(X), np.array(y), timestamps
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """テクニカル指標を追加
        
        Args:
            df: 為替レートのデータフレーム
        """
        if self.features is None:
            return
        
        # SMA (Simple Moving Average)
        if "sma_5" in self.features:
            df["sma_5"] = df["close"].rolling(5).mean()
        if "sma_10" in self.features:
            df["sma_10"] = df["close"].rolling(10).mean()
        if "sma_20" in self.features:
            df["sma_20"] = df["close"].rolling(20).mean()
        
        # EMA (Exponential Moving Average)
        if "ema_5" in self.features:
            df["ema_5"] = df["close"].ewm(span=5).mean()
        if "ema_10" in self.features:
            df["ema_10"] = df["close"].ewm(span=10).mean()
        if "ema_20" in self.features:
            df["ema_20"] = df["close"].ewm(span=20).mean()
        
        # ボリンジャーバンド
        if any(f in self.features for f in ["bb_upper", "bb_middle", "bb_lower"]):
            window = 20
            std = df["close"].rolling(window).std()
            df["bb_middle"] = df["close"].rolling(window).mean()
            df["bb_upper"] = df["bb_middle"] + 2 * std
            df["bb_lower"] = df["bb_middle"] - 2 * std
        
        # RSI (Relative Strength Index)
        if "rsi" in self.features:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        if any(f in self.features for f in ["macd", "macd_signal", "macd_hist"]):
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # 変化率
        if "pct_change" in self.features:
            df["pct_change"] = df["close"].pct_change()
        
        # ATR (Average True Range)
        if "atr" in self.features:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["atr"] = true_range.rolling(14).mean()
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """モデルの構築
        
        Args:
            input_shape: 入力データの形状 (sequence_length, features)
        """
        model = tf.keras.models.Sequential()
        
        # 入力層
        model.add(tf.keras.layers.LSTM(
            units=self.units,
            return_sequences=self.n_layers > 1,
            input_shape=input_shape
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # 中間層
        for i in range(1, self.n_layers):
            is_last_layer = i == self.n_layers - 1
            model.add(tf.keras.layers.LSTM(
                units=self.units,
                return_sequences=not is_last_layer
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # 出力層
        model.add(tf.keras.layers.Dense(1))
        
        # コンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error"
        )
        
        self.model = model
        logger.info(f"LSTMモデルを構築しました: レイヤー数={self.n_layers}, ユニット数={self.units}")
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        target_col: str = "close",
        early_stopping: bool = True,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """モデルの学習
        
        Args:
            df_train: 学習用データ
            df_val: 検証用データ
            target_col: 予測対象の列名
            early_stopping: 早期終了を使用するかどうか
            save_model: 学習後にモデルを保存するかどうか
            
        Returns:
            学習結果の情報
        """
        # 学習開始時間
        training_start = datetime.datetime.utcnow()
        
        # データの前処理
        X_train, y_train, _ = self._preprocess_data(df_train, target_col)
        
        # 検証データがあれば前処理
        validation_data = None
        if df_val is not None:
            X_val, y_val, _ = self._preprocess_data(df_val, target_col)
            validation_data = (X_val, y_val)
        
        # モデルの構築
        if self.model is None:
            self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # コールバックの設定
        callbacks = []
        
        if early_stopping:
            # 早期終了
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=10,
                restore_best_weights=True
            ))
        
        # 学習率スケジューラ
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5
        ))
        
        # モデルチェックポイント
        if save_model:
            os.makedirs(self.app_config.save_models_path, exist_ok=True)
            model_save_path = os.path.join(
                self.app_config.save_models_path,
                f"lstm_{self.currency_pair}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
            
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                model_save_path,
                save_best_only=True,
                monitor="val_loss" if validation_data else "loss"
            ))
        
        # 学習の実行
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # 学習終了時間
        training_end = datetime.datetime.utcnow()
        training_duration = (training_end - training_start).total_seconds()
        
        # 学習結果
        train_metrics = {
            "loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1] if validation_data else None,
            "training_start": training_start,
            "training_end": training_end,
            "training_duration": training_duration,
            "model_path": model_save_path if save_model else None
        }
        
        logger.info(f"モデル学習完了: 損失={train_metrics['loss']:.6f}, "
                   f"検証損失={train_metrics['val_loss']:.6f if train_metrics['val_loss'] else 'N/A'}")
        
        # データベースに保存
        if save_model:
            self._save_model_metadata(train_metrics)
        
        return train_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        target_col: str = "close"
    ) -> Tuple[np.ndarray, List[datetime.datetime]]:
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
        X, _, timestamps = self._preprocess_data(df, target_col)
        
        # 予測の実行
        y_pred_scaled = self.model.predict(X)
        
        # スケール逆変換
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred, timestamps
    
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
        X, y_true_scaled, timestamps = self._preprocess_data(df, target_col)
        
        # 予測の実行
        y_pred_scaled = self.model.predict(X)
        
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
            "timestamp": datetime.datetime.utcnow()
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
                f"lstm_{self.currency_pair}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # モデル本体の保存
        model_path = f"{path}.h5"
        self.model.save(model_path)
        
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
        self.model = tf.keras.models.load_model(model_path)
        
        # 設定の読み込み
        if config_path is None:
            config_path = model_path.replace(".h5", "_config.joblib")
        
        if os.path.exists(config_path):
            loaded_config = joblib.load(config_path)
            self.hyperparameters = loaded_config.get("hyperparameters", self.hyperparameters)
            self.model_type = loaded_config.get("model_type", self.model_type)
            self.currency_pair = loaded_config.get("currency_pair", self.currency_pair)
            self.features = loaded_config.get("features", self.features)
            self.sequence_length = loaded_config.get("sequence_length", self.sequence_length)
        
        # スケーラーの読み込み
        if scaler_path is None:
            scaler_path = model_path.replace(".h5", "_scalers.joblib")
        
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler_X = scalers.get("scaler_X")
            self.scaler_y = scalers.get("scaler_y")
        
        logger.info(f"モデルを読み込みました: {model_path}")
    
    def _save_model_metadata(self, train_metrics: Dict[str, Any]) -> None:
        """モデルのメタデータをデータベースに保存
        
        Args:
            train_metrics: 学習結果
        """
        try:
            session = self.db.get_session()
            
            # モデルの保存
            db_model = DbModel(
                uuid=self.model_uuid or str(uuid.uuid4()),
                name=self.config.get("name", f"LSTM_{self.currency_pair}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"),
                model_type=self.model_type,
                currency_pair=self.currency_pair,
                generation=self.config.get("generation", 0),
                hyperparameters=self.hyperparameters,
                model_path=train_metrics.get("model_path"),
                features=self.features,
                training_start=train_metrics.get("training_start"),
                training_end=train_metrics.get("training_end"),
                training_duration=train_metrics.get("training_duration")
            )
            
            # 親モデルがある場合は設定
            parent_uuid = self.config.get("parent_uuid")
            if parent_uuid:
                if isinstance(parent_uuid, list):
                    # 複数の親がある場合は最初の親を設定
                    db_model.parent_uuid = parent_uuid[0]
                else:
                    db_model.parent_uuid = parent_uuid
            
            session.add(db_model)
            session.commit()
            
            # モデルIDを保存
            self.db_model_id = db_model.id
            
            logger.info(f"モデルメタデータをデータベースに保存しました: ID={db_model.id}")
            
        except Exception as e:
            logger.error(f"モデルメタデータの保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close()
    
    def _save_evaluation_results(
        self,
        evaluation: Dict[str, Any],
        start_date: datetime.datetime,
        end_date: datetime.datetime
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
                
                # 精度指標
                accuracy=evaluation.get("direction_accuracy", 0) * 100,  # パーセンテージに変換
                
                # 詳細情報
                details={
                    "mse": evaluation.get("mse"),
                    "rmse": evaluation.get("rmse"),
                    "mae": evaluation.get("mae")
                }
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
        timestamps: List[datetime.datetime]
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
            prediction_time = datetime.datetime.utcnow()
            
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
                    confidence=0.5,  # LSTMではデフォルトで0.5とする
                    
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