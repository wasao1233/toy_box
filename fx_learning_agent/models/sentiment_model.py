import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import uuid

from utils.logger import get_logger
from config.config import get_config
from models.data_models import Model as DbModel, ModelPerformance, NewsItem
from utils.database import get_db

logger = get_logger(__name__)

class SentimentModel:
    """ニュースセンチメント分析モデル
    
    ニュース記事のセンチメント（感情極性）を分析し、
    市場への影響度を予測するモデル。
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
        self.model_type = self.config.get("model_type", "BERT")
        self.currency_pair = self.config.get("currency_pair", "USD/JPY")
        
        # ハイパーパラメータ
        self.hyperparameters = self.config.get("hyperparameters", {})
        
        # デフォルトのハイパーパラメータ
        self.pretrained_model = self.hyperparameters.get(
            "pretrained_model", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.max_length = self.hyperparameters.get("max_length", 512)
        self.batch_size = self.hyperparameters.get("batch_size", 16)
        self.learning_rate = self.hyperparameters.get("learning_rate", 5e-5)
        self.weight_decay = self.hyperparameters.get("weight_decay", 0.01)
        self.epochs = self.hyperparameters.get("epochs", 3)
        
        # モデルオブジェクト
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルID（データベース保存用）
        self.db_model_id = None
        self.model_uuid = self.config.get("uuid")
        
        # ラベルマッピング
        self.id2label = {0: "negative", 1: "positive"}
        self.label2id = {"negative": 0, "positive": 1}
    
    def initialize(self):
        """モデルの初期化"""
        try:
            # トークナイザーのロード
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            
            # モデルのロード
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model,
                num_labels=2,
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # モデルをデバイスに移動
            self.model = self.model.to(self.device)
            logger.info(f"センチメントモデルを初期化しました: {self.pretrained_model} (デバイス: {self.device})")
            
        except Exception as e:
            logger.error(f"センチメントモデルの初期化エラー: {str(e)}", exc_info=True)
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """テキストの前処理
        
        Args:
            texts: テキストのリスト
            
        Returns:
            トークン化されたテキスト
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
    
    def predict_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """センチメントの予測
        
        Args:
            texts: テキストのリスト
            
        Returns:
            センチメント予測結果のリスト
        """
        if self.model is None:
            self.initialize()
        
        # 予測結果
        results = []
        
        # バッチ処理
        batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 空のテキストをスキップ
            batch_texts = [text for text in batch_texts if text and len(text) > 10]
            if not batch_texts:
                continue
            
            # テキストの前処理
            inputs = self._preprocess_texts(batch_texts)
            
            # 推論モードに設定
            self.model.eval()
            
            # 勾配計算を無効化
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # ロジットからセンチメントスコアを計算
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # センチメントスコアを-1から1の範囲に変換
            # 0は完全にネガティブ、1は完全にポジティブを表す
            sentiment_scores = probabilities[:, 1].cpu().numpy()
            sentiment_scores = 2 * sentiment_scores - 1  # 0-1 から -1-1 の範囲に変換
            
            for text, score in zip(batch_texts, sentiment_scores):
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment_score": float(score),
                    "sentiment": "positive" if score > 0 else "negative",
                    "confidence": abs(score)  # 信頼度は絶対値
                })
        
        return results
    
    def analyze_news_sentiment(
        self,
        news_items: List[Dict[str, Any]],
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """ニュース記事のセンチメント分析
        
        Args:
            news_items: ニュース記事のリスト
            save_to_db: データベースに保存するかどうか
            
        Returns:
            センチメント分析結果のデータフレーム
        """
        if not news_items:
            logger.warning("分析するニュース記事がありません")
            return pd.DataFrame()
        
        # ニュース記事のテキスト抽出
        texts = []
        news_ids = []
        
        for news in news_items:
            # タイトルと要約を結合（要約がない場合はタイトルのみ）
            text = news.get("title", "")
            if news.get("summary"):
                text = text + ". " + news.get("summary")
            
            texts.append(text)
            news_ids.append(news.get("id"))
        
        # センチメント予測
        predictions = self.predict_sentiment(texts)
        
        # 結果をデータフレームに変換
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "news_id": news_ids[i] if i < len(news_ids) else None,
                "sentiment_score": pred["sentiment_score"],
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
                "timestamp": datetime.datetime.utcnow()
            })
        
        df_results = pd.DataFrame(results)
        
        # データベースに保存
        if save_to_db and not df_results.empty:
            self._save_sentiment_to_db(df_results)
        
        return df_results
    
    def _save_sentiment_to_db(self, df_results: pd.DataFrame) -> None:
        """センチメント分析結果をデータベースに保存
        
        Args:
            df_results: センチメント分析結果のデータフレーム
        """
        try:
            session = self.db.get_session()
            count = 0
            
            for _, row in df_results.iterrows():
                # ニュース記事の取得
                news_id = row.get("news_id")
                if news_id is None:
                    continue
                
                news_item = session.query(NewsItem).filter(NewsItem.id == news_id).first()
                if news_item:
                    # センチメントスコアの更新
                    news_item.sentiment_score = row["sentiment_score"]
                    count += 1
            
            session.commit()
            logger.info(f"{count}件のセンチメントスコアをデータベースに保存しました")
            
        except Exception as e:
            logger.error(f"センチメントスコアの保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close()
    
    def calculate_market_sentiment(
        self,
        currency_pair: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """市場センチメントの計算
        
        Args:
            currency_pair: 通貨ペア
            start_date: 開始日時
            end_date: 終了日時
            
        Returns:
            市場センチメントの情報
        """
        if currency_pair is None:
            currency_pair = self.currency_pair
        
        if end_date is None:
            end_date = datetime.datetime.utcnow()
        
        if start_date is None:
            start_date = end_date - datetime.timedelta(days=7)
        
        try:
            session = self.db.get_session()
            
            # 指定された通貨ペアに関連するニュース記事を取得
            query = session.query(NewsItem).filter(
                NewsItem.published_at >= start_date,
                NewsItem.published_at <= end_date,
                NewsItem.sentiment_score != None  # センチメントスコアが計算済みの記事のみ
            )
            
            if currency_pair:
                query = query.filter(NewsItem.related_currencies.like(f"%{currency_pair}%"))
            
            news_items = query.all()
            
            if not news_items:
                logger.warning(f"期間内のニュース記事がありません: {start_date} から {end_date}")
                return {
                    "currency_pair": currency_pair,
                    "start_date": start_date,
                    "end_date": end_date,
                    "avg_sentiment": 0,
                    "sentiment_direction": "neutral",
                    "news_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "calculation_time": datetime.datetime.utcnow()
                }
            
            # センチメントスコアの集計
            sentiment_scores = [item.sentiment_score for item in news_items]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # ポジティブ/ネガティブ記事のカウント
            positive_count = sum(1 for score in sentiment_scores if score > 0)
            negative_count = sum(1 for score in sentiment_scores if score < 0)
            
            # センチメント方向
            if avg_sentiment > 0.1:
                sentiment_direction = "positive"
            elif avg_sentiment < -0.1:
                sentiment_direction = "negative"
            else:
                sentiment_direction = "neutral"
            
            # 市場センチメント情報
            market_sentiment = {
                "currency_pair": currency_pair,
                "start_date": start_date,
                "end_date": end_date,
                "avg_sentiment": avg_sentiment,
                "sentiment_direction": sentiment_direction,
                "news_count": len(news_items),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "calculation_time": datetime.datetime.utcnow()
            }
            
            logger.info(f"市場センチメント計算: {currency_pair}, "
                       f"平均={avg_sentiment:.4f}, 方向={sentiment_direction}, "
                       f"記事数={len(news_items)}")
            
            return market_sentiment
            
        except Exception as e:
            logger.error(f"市場センチメント計算エラー: {str(e)}", exc_info=True)
            return {
                "currency_pair": currency_pair,
                "error": str(e),
                "calculation_time": datetime.datetime.utcnow()
            }
        finally:
            session.close()
    
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
                f"sentiment_{self.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # モデル本体の保存
        model_path = f"{path}_model"
        self.model.save_pretrained(model_path)
        
        # トークナイザーの保存
        tokenizer_path = f"{path}_tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # 設定の保存
        config_path = f"{path}_config.joblib"
        config_to_save = {
            "hyperparameters": self.hyperparameters,
            "model_type": self.model_type,
            "currency_pair": self.currency_pair,
            "id2label": self.id2label,
            "label2id": self.label2id
        }
        joblib.dump(config_to_save, config_path)
        
        logger.info(f"センチメントモデルを保存しました: {model_path}")
        
        return model_path
    
    def load(self, model_path: str, tokenizer_path: Optional[str] = None, config_path: Optional[str] = None) -> None:
        """モデルの読み込み
        
        Args:
            model_path: モデルディレクトリのパス
            tokenizer_path: トークナイザーディレクトリのパス
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        if config_path is None:
            config_path = model_path.replace("_model", "_config.joblib")
        
        if os.path.exists(config_path):
            loaded_config = joblib.load(config_path)
            self.hyperparameters = loaded_config.get("hyperparameters", self.hyperparameters)
            self.model_type = loaded_config.get("model_type", self.model_type)
            self.currency_pair = loaded_config.get("currency_pair", self.currency_pair)
            self.id2label = loaded_config.get("id2label", self.id2label)
            self.label2id = loaded_config.get("label2id", self.label2id)
        
        # トークナイザーの読み込み
        if tokenizer_path is None:
            tokenizer_path = model_path.replace("_model", "_tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # モデルの読み込み
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # モデルをデバイスに移動
        self.model = self.model.to(self.device)
        
        logger.info(f"センチメントモデルを読み込みました: {model_path}")
    
    def _save_model_metadata(self) -> None:
        """モデルのメタデータをデータベースに保存"""
        try:
            session = self.db.get_session()
            
            # モデルの保存
            db_model = DbModel(
                uuid=self.model_uuid or str(uuid.uuid4()),
                name=self.config.get("name", f"Sentiment_{self.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"),
                model_type=self.model_type,
                currency_pair=self.currency_pair,
                generation=self.config.get("generation", 0),
                hyperparameters=self.hyperparameters,
                model_path=None,  # 実際のモデルパスは後で更新
                features=None
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
            
            logger.info(f"センチメントモデルのメタデータをデータベースに保存しました: ID={db_model.id}")
            
        except Exception as e:
            logger.error(f"モデルメタデータの保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close() 