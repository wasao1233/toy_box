import datetime
import json
import uuid
from typing import Optional, List, Dict, Any, Union
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from utils.database import Base

class CurrencyRate(Base):
    """為替レートのデータモデル"""
    __tablename__ = "currency_rates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    timeframe = Column(String(10), nullable=False, index=True)  # 1m, 5m, 15m, 1h, 4h, 1d など
    source = Column(String(50), nullable=False)  # データソース（Alpha Vantage, YFなど）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<CurrencyRate(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class NewsItem(Base):
    """ニュース記事のデータモデル"""
    __tablename__ = "news_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=False)
    url = Column(String(1000), nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=True)  # -1.0 (非常にネガティブ) から 1.0 (非常にポジティブ)
    categories = Column(String(500), nullable=True)  # カンマ区切りのカテゴリリスト
    related_currencies = Column(String(200), nullable=True)  # 関連する通貨ペア
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<NewsItem(title='{self.title[:30]}...', published_at='{self.published_at}')>"


class Model(Base):
    """学習モデルのメタデータ"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), nullable=False, unique=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # LSTM, ARIMA, NLP, Hybrid など
    currency_pair = Column(String(20), nullable=False, index=True)
    generation = Column(Integer, nullable=False, default=0)  # 世代番号
    parent_id = Column(Integer, ForeignKey("models.id"), nullable=True)  # 親モデルID
    parent_uuid = Column(String(36), nullable=True)  # 親モデルUUID
    hyperparameters = Column(JSON, nullable=False)  # ハイパーパラメータ
    model_path = Column(String(500), nullable=True)  # モデルファイルのパス
    features = Column(JSON, nullable=True)  # 使用する特徴量
    training_start = Column(DateTime, nullable=True)
    training_end = Column(DateTime, nullable=True)
    training_duration = Column(Float, nullable=True)  # 学習時間（秒）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # リレーションシップ
    children = relationship("Model", backref="parent", remote_side=[id])
    performances = relationship("ModelPerformance", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")
    trades = relationship("BacktestTrade", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Model(name='{self.name}', uuid='{self.uuid}', generation={self.generation})>"


class ModelPerformance(Base):
    """モデルのパフォーマンス評価"""
    __tablename__ = "model_performances"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    evaluation_type = Column(String(20), nullable=False)  # training, validation, test
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # パフォーマンス指標
    profit = Column(Float, nullable=False, default=0.0)  # 利益（通貨単位）
    profit_percent = Column(Float, nullable=False, default=0.0)  # 利益率（%）
    win_count = Column(Integer, nullable=False, default=0)  # 勝ちトレード数
    loss_count = Column(Integer, nullable=False, default=0)  # 負けトレード数
    win_rate = Column(Float, nullable=False, default=0.0)  # 勝率（%）
    avg_win = Column(Float, nullable=True)  # 平均勝ちトレード（通貨単位）
    avg_loss = Column(Float, nullable=True)  # 平均負けトレード（通貨単位）
    max_drawdown = Column(Float, nullable=True)  # 最大ドローダウン（%）
    sharpe_ratio = Column(Float, nullable=True)  # シャープレシオ
    sortino_ratio = Column(Float, nullable=True)  # ソルティノレシオ
    
    # 主要な精度指標
    accuracy = Column(Float, nullable=True)  # 方向予測の正確さ（%）
    precision = Column(Float, nullable=True)  # 精度（正の予測の正確さ）
    recall = Column(Float, nullable=True)  # 再現率
    f1_score = Column(Float, nullable=True)  # F1スコア
    
    # フィットネススコア（進化的アルゴリズム用）
    fitness_score = Column(Float, nullable=False, default=0.0)
    
    # 追加情報
    details = Column(JSON, nullable=True)  # 詳細な評価指標
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # リレーションシップ
    model = relationship("Model", back_populates="performances")
    
    def __repr__(self):
        return f"<ModelPerformance(model_id={self.model_id}, profit={self.profit}, win_rate={self.win_rate})>"


class Prediction(Base):
    """モデルの予測結果"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)  # 予測を行った時刻
    target_timestamp = Column(DateTime, nullable=False)  # 予測対象の時刻
    symbol = Column(String(20), nullable=False)
    
    # 予測値
    predicted_direction = Column(String(10), nullable=False)  # up, down, neutral
    predicted_change = Column(Float, nullable=True)  # 予測される変化率（%）
    predicted_price = Column(Float, nullable=True)  # 予測価格
    confidence = Column(Float, nullable=True)  # 予測の信頼度（0-1）
    
    # 実際の値（結果判明後に更新）
    actual_direction = Column(String(10), nullable=True)
    actual_change = Column(Float, nullable=True)
    actual_price = Column(Float, nullable=True)
    
    # 結果評価
    is_correct = Column(Boolean, nullable=True)
    error = Column(Float, nullable=True)  # 予測誤差
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # リレーションシップ
    model = relationship("Model", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(model_id={self.model_id}, symbol='{self.symbol}', target_timestamp='{self.target_timestamp}')>"


class BacktestTrade(Base):
    """バックテスト取引のデータモデル"""
    __tablename__ = "backtest_trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # buy, sell
    
    # 取引タイミング
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    
    # 取引量と損益
    volume = Column(Float, nullable=False)
    profit_loss = Column(Float, nullable=True)
    profit_loss_pips = Column(Float, nullable=True)
    profit_loss_percent = Column(Float, nullable=True)
    
    # リスク管理
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # 取引理由と追加情報
    entry_reason = Column(String(500), nullable=True)
    exit_reason = Column(String(500), nullable=True)
    additional_info = Column(JSON, nullable=True)
    
    status = Column(String(20), nullable=False, default="open")  # open, closed, cancelled
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # リレーションシップ
    model = relationship("Model", back_populates="trades")
    
    def __repr__(self):
        return f"<BacktestTrade(model_id={self.model_id}, symbol='{self.symbol}', status='{self.status}')>" 