#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import signal
import logging
from datetime import datetime, timedelta
import uuid
import json
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 相対インポートのためにプロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_config
from config.model_config import ModelConfig, get_lstm_config, get_sentiment_config
from utils.logger import get_logger
from utils.database import get_db
from data.data_fetcher import get_forex_fetcher, get_news_fetcher
from models.evolution import GeneticAlgorithm
from models.lstm_model import TimeSeriesModel
from models.sentiment_model import SentimentModel
from models.data_models import Model as DbModel

# ロガーの設定
logger = get_logger("main")

class FXLearningAgent:
    """FX学習エージェント
    
    為替市場のデータとニュースを解析し、取引戦略を学習するエージェント。
    過去のデータを利用した取引デモ、そして進化的手法を活用して世代ごとに
    成績を向上させることを目指す。
    """
    
    def __init__(self, num_workers: int = 15):
        """初期化"""
        self.num_workers = num_workers
        self.db = get_db()
        self.data_fetcher = get_forex_fetcher()
        self.sentiment_model = None  # 初期化時にNoneを設定
        self.logger = get_logger(__name__)  # logger属性を追加
        self.currency_pair = None  # 通貨ペアの初期化
        
        self.logger.info(f"FX学習エージェントを初期化しました (ワーカー数: {num_workers})")
        
    def _get_model(self, model_type: str, currency_pair: str = None) -> Any:
        """モデルのインスタンスを取得
        
        Args:
            model_type: モデルタイプ ("lstm", "sentiment", "randomforest")
            currency_pair: 通貨ペア
            
        Returns:
            モデルのインスタンス
        """
        model_type = model_type.lower()
        
        if model_type == "lstm":
            from models.lstm_model import TimeSeriesModel
            return TimeSeriesModel(currency_pair=currency_pair)
        elif model_type == "sentiment":
            from models.sentiment_model import SentimentModel
            return SentimentModel()
        elif model_type == "randomforest":
            from models.lstm_model import TimeSeriesModel
            return TimeSeriesModel(currency_pair=currency_pair)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
    
    def initialize(self):
        """FX学習エージェントを初期化します"""
        try:
            if self.sentiment_model is not None:
                self.sentiment_model.initialize()
        except Exception as e:
            self.logger.error(f"センチメントモデルの初期化エラー: {e}")
        
        self.logger.info("FX学習エージェントの初期化が完了しました")
    
    def fetch_data(self, currency_pair: str, timeframe: str, days: int = 30):
        """データ取得
        
        Args:
            currency_pair: 通貨ペア
            timeframe: 時間枠
            days: 取得日数
        """
        try:
            self.logger.info(f"為替データの取得: {currency_pair}, {timeframe}, {days}日間")
            
            # 日付範囲の設定
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # 為替データの取得
            df = self.data_fetcher.fetch_historical_rates(
                symbol=currency_pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                save_to_db=True
            )
            
            if df.empty:
                self.logger.warning(f"為替データが取得できませんでした: {currency_pair}, {timeframe}")
                return None
            
            self.logger.info(f"{len(df)}件の為替データを取得しました: {currency_pair}, {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"データ取得エラー: {str(e)}", exc_info=True)
            return None
    
    def fetch_news(self, keywords: List[str], days: int = 7):
        """ニュースデータ取得
        
        Args:
            keywords: 検索キーワード
            days: 取得日数
        """
        try:
            self.logger.info(f"ニュースデータの取得: {keywords}, {days}日間")
            
            # デフォルトのキーワードを拡張
            if not keywords:
                currency_pairs = [cp.symbol for cp in self.config.currency_pairs]
                keywords = currency_pairs + [
                    # 金融政策関連
                    "FRB", "ECB", "BOJ", "日銀", "利上げ", "利下げ", "金融政策",
                    # 経済指標
                    "GDP", "失業率", "インフレ率", "CPI", "PPI",
                    # 地政学リスク
                    "トランプ", "関税", "貿易摩擦", "地政学リスク",
                    # 市場指数
                    "VIX", "S&P500", "ダウ平均", "日経平均", "TOPIX"
                ]
            
            # 日付範囲の設定
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # ニュースデータの取得
            articles = self.news_fetcher.fetch_news(
                keywords=keywords,
                from_date=start_date,
                to_date=end_date,
                save_to_db=True
            )
            
            if not articles:
                self.logger.warning(f"ニュースデータが取得できませんでした: {keywords}")
                return None
            
            self.logger.info(f"{len(articles)}件のニュースデータを取得しました")
            
            # 市場指数データの取得
            market_indices = self.data_fetcher.fetch_market_indices(
                timeframe="1d",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_indices:
                self.logger.info(f"{len(market_indices)}件の市場指数データを取得しました")
            
            # センチメント分析
            if hasattr(self, 'sentiment_model') and self.sentiment_model.model is not None:
                self.sentiment_model.analyze_news_sentiment(articles)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"ニュース取得エラー: {str(e)}", exc_info=True)
            return None
    
    def train(self, currency_pair: str, model_type: str = "lstm", **kwargs):
        """モデルの学習を実行"""
        try:
            # モデルの初期化
            if model_type.lower() == "lstm":
                self.model = TimeSeriesModel(currency_pair=currency_pair)
            else:
                raise ValueError(f"未対応のモデルタイプです: {model_type}")
            
            # データの取得
            self.logger.info(f"1年分のデータを取得します: {currency_pair}")
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
            
            # 為替データの取得
            rates = self.data_fetcher.fetch_historical_rates(
                symbol=currency_pair,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
            
            if rates.empty:
                raise ValueError("学習データを取得できませんでした")
            
            self.logger.info(f"{len(rates)}件の為替データを取得しました: {currency_pair}, 1d")
            
            # 市場指数データの取得
            market_indices = self.data_fetcher.fetch_market_indices(
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
            self.logger.info(f"市場指数データを取得しました: {list(market_indices.keys())}")
            
            # データの整合性を確保
            dates = rates.index
            for name, df in market_indices.items():
                # 日付でフィルタリング
                df = df[df.index.isin(dates)]
                # 欠損値の補完
                df = df.reindex(dates)
                df = df.fillna(method='ffill').fillna(method='bfill')
                market_indices[name] = df
            
            # データの分割
            train_size = int(len(rates) * 0.8)
            train_data = rates[:train_size]
            val_data = rates[train_size:]
            
            self.logger.info(f"学習データ: {len(train_data)}件, 検証データ: {len(val_data)}件")
            
            # モデルの学習
            self.logger.info(f"{model_type}モデルの学習を開始します...")
            training_start = datetime.now()
            self.model.fit(
                data=train_data,
                validation_data=val_data,
                market_indices=market_indices
            )
            training_end = datetime.now()
            
            # モデルの評価
            val_mse, val_rmse, val_mae = self.model.evaluate(val_data, market_indices=market_indices)
            
            # モデルの保存
            model_saved = self.model.save()
            
            # 学習結果の記録
            training_duration = (training_end - training_start).total_seconds()
            result = {
                'training_start': training_start,
                'training_end': training_end,
                'training_duration': training_duration,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'model_saved': model_saved,
                'mse': val_mse,
                'rmse': val_rmse,
                'mae': val_mae,
                'direction_accuracy': self.model.direction_accuracy,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"モデル学習が完了しました。モデルID: {self.model.model_uuid}")
            self.logger.info(f"学習結果: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"モデル学習中にエラーが発生しました: {str(e)}")
            raise
    
    def run_evolution(
        self,
        currency_pair: str,
        model_type: str = "LSTM",
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        parallel: bool = True
    ):
        """進化的アルゴリズムの実行
        
        Args:
            currency_pair: 通貨ペア
            model_type: モデルタイプ
            generations: 世代数
            population_size: 集団サイズ
            parallel: 並列計算を行うかどうか
        """
        try:
            # 設定の取得
            if generations is None:
                generations = self.config.learning.generations
            
            if population_size is None:
                population_size = self.config.learning.initial_population
            
            self.logger.info(f"進化的学習の開始: {currency_pair}, {model_type}, "
                       f"世代数={generations}, 集団サイズ={population_size}")
            
            # データの取得
            train_data = self.fetch_data(currency_pair, "1h", days=60)
            val_data = self.fetch_data(currency_pair, "1h", days=10)
            
            if train_data is None or val_data is None:
                self.logger.error("学習データが取得できませんでした")
                return
            
            # ハイパーパラメータ探索空間の定義
            if model_type == "LSTM":
                hyperparameter_space = {
                    "sequence_length": {
                        "type": "int",
                        "min": 20,
                        "max": 100
                    },
                    "n_layers": {
                        "type": "int",
                        "min": 1,
                        "max": 3
                    },
                    "units": {
                        "type": "int",
                        "min": 32,
                        "max": 128,
                        "log_scale": True
                    },
                    "dropout_rate": {
                        "type": "float",
                        "min": 0.1,
                        "max": 0.5
                    },
                    "learning_rate": {
                        "type": "float",
                        "min": 1e-4,
                        "max": 1e-2,
                        "log_scale": True
                    },
                    "batch_size": {
                        "type": "int",
                        "min": 16,
                        "max": 64,
                        "log_scale": True
                    },
                    "epochs": {
                        "type": "int",
                        "min": 20,
                        "max": 100
                    }
                }
                
                # 特徴量の定義
                all_features = [
                    "open", "high", "low", "close", "volume",
                    "sma_5", "sma_10", "sma_20",
                    "ema_5", "ema_10", "ema_20",
                    "bb_upper", "bb_middle", "bb_lower",
                    "rsi", "macd", "macd_signal", "macd_hist",
                    "pct_change", "atr"
                ]
                
            else:
                self.logger.error(f"サポートされていないモデルタイプ: {model_type}")
                return
            
            # 初期集団の生成
            population = self.genetic_algorithm.initialize_population(
                model_type=model_type,
                currency_pair=currency_pair,
                hyperparameter_space=hyperparameter_space,
                features=all_features
            )
            
            # 世代ループ
            for generation in range(generations):
                self.logger.info(f"世代 {generation+1}/{generations} の学習開始")
                
                # 各モデルの学習と評価（並列処理）
                train_results = []
                
                if parallel and self.num_workers > 1:
                    # 並列処理
                    with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = []
                        for model_config in population:
                            future = executor.submit(
                                self.train,
                                currency_pair,
                                model_type,
                                **model_config
                            )
                            futures.append(future)
                        
                        # 結果の収集
                        for future in futures:
                            train_results.append(future.result())
                else:
                    # 順次処理
                    for model_config in population:
                        result = self.train(currency_pair, model_type, **model_config)
                        train_results.append(result)
                
                # パフォーマンス情報の抽出
                performances = []
                
                for result in train_results:
                    if result["status"] == "success":
                        perf = {
                            "model_uuid": result["model_uuid"],
                            "model_id": result["model_id"]
                        }
                        
                        # 評価指標の追加
                        if "results" in result:
                            res = result["results"]
                            perf.update({
                                "loss": res.get("loss", 0),
                                "val_loss": res.get("val_loss", 0),
                                "direction_accuracy": res.get("direction_accuracy", 0),
                                "mae": res.get("mae", 0),
                                "rmse": res.get("rmse", 0)
                            })
                        
                        performances.append(perf)
                
                # 次世代の生成
                if generation < generations - 1:  # 最終世代では次世代を生成しない
                    next_population = self.genetic_algorithm.evolve_generation(
                        current_population=population,
                        performances=performances,
                        hyperparameter_space=hyperparameter_space,
                        all_features=all_features
                    )
                    
                    # 世代の更新
                    population = next_population
                
                self.logger.info(f"世代 {generation+1}/{generations} の学習完了")
            
            self.logger.info(f"進化的学習完了: {generations}世代, {len(population)}モデル")
            
        except Exception as e:
            self.logger.error(f"進化的学習エラー: {str(e)}", exc_info=True)
    
    def run_backtest(self, model_id: int, test_data: Any = None):
        """バックテストの実行
        
        Args:
            model_id: モデルID
            test_data: テストデータ（指定がなければ新たに取得）
        """
        # TODO: バックテストの実装
        self.logger.warning("バックテスト機能は未実装です")
    
    def shutdown(self):
        """エージェントのシャットダウン"""
        self.logger.info("FX学習エージェントをシャットダウンしています...")
        self.shutdown_requested = True
        
        # データベース接続のクローズ
        if hasattr(self, 'db') and self.db:
            self.db.dispose()
        
        self.logger.info("FX学習エージェントをシャットダウンしました")

    def evaluate(self, model_id: int) -> Dict[str, float]:
        """モデルの評価を実行
        
        Args:
            model_id: 評価対象のモデルID
            
        Returns:
            評価指標の辞書
        """
        logger.info(f"モデル（ID: {model_id}）の評価を開始")
        
        # テストデータの取得
        test_data, market_indices = self._get_test_data()
        
        # モデルのロード
        self.model.load_model(model_id)
        
        # 予測の実行
        predictions, timestamps = self.model.predict(test_data, market_indices=market_indices)
        
        # 実際の値の取得
        actual = test_data["close"].values.reshape(-1, 1)
        
        # 評価指標の計算
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
        
        logger.info(f"評価結果: {metrics}")
        return metrics

    def _get_test_data(self):
        """テストデータを取得します。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特徴量とターゲットのデータフレーム
        """
        self.logger.info("テストデータを取得しています...")
        
        # 為替データを取得
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)  # 過去30日分のデータを使用
        
        forex_data = self.data_fetcher.fetch_historical_rates(
            symbol=self.currency_pair,
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        # 市場指数データを取得
        market_indices = self.data_fetcher.fetch_market_indices(
            indices=["VIX", "S&P500", "日経平均", "DXY", "金", "原油"],
            start_date=start_date,
            end_date=end_date
        )
        
        self.logger.info(f"テストデータを取得しました: {len(forex_data)}件")
        
        return forex_data, market_indices


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="FX学習エージェント")
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # データ取得コマンド
    fetch_parser = subparsers.add_parser("fetch", help="データ取得")
    fetch_parser.add_argument("--pair", "-p", type=str, default="USD/JPY", help="通貨ペア")
    fetch_parser.add_argument("--timeframe", "-t", type=str, default="1h", help="時間枠")
    fetch_parser.add_argument("--days", "-d", type=int, default=30, help="取得日数")
    
    # ニュース取得コマンド
    news_parser = subparsers.add_parser("news", help="ニュース取得")
    news_parser.add_argument("--keywords", "-k", type=str, nargs="+", help="検索キーワード")
    news_parser.add_argument("--days", "-d", type=int, default=7, help="取得日数")
    
    # モデル学習コマンド
    train_parser = subparsers.add_parser("train", help="モデル学習")
    train_parser.add_argument("--pair", "-p", type=str, default="USD/JPY", help="通貨ペア")
    train_parser.add_argument("--model", "-m", type=str, default="LSTM", help="モデルタイプ")
    
    # 進化的学習コマンド
    evolve_parser = subparsers.add_parser("evolve", help="進化的学習")
    evolve_parser.add_argument("--pair", "-p", type=str, default="USD/JPY", help="通貨ペア")
    evolve_parser.add_argument("--model", "-m", type=str, default="LSTM", help="モデルタイプ")
    evolve_parser.add_argument("--generations", "-g", type=int, help="世代数")
    evolve_parser.add_argument("--population", "-n", type=int, help="集団サイズ")
    evolve_parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="並列処理を無効化")
    
    # バックテストコマンド
    backtest_parser = subparsers.add_parser("backtest", help="バックテスト")
    backtest_parser.add_argument("--model-id", "-i", type=int, required=True, help="モデルID")
    
    # コマンドライン引数の解析
    args = parser.parse_args()
    
    # FX学習エージェントの初期化
    agent = FXLearningAgent()
    agent.initialize()
    
    try:
        # コマンドの実行
        if args.command == "fetch":
            agent.fetch_data(args.pair, args.timeframe, args.days)
            
        elif args.command == "news":
            if not args.keywords:
                # デフォルトのキーワード
                currency_pairs = [cp.symbol for cp in agent.config.currency_pairs]
                keywords = currency_pairs + ["FRB", "ECB", "BOJ", "inflation", "interest rate"]
            else:
                keywords = args.keywords
                
            agent.fetch_news(keywords, args.days)
            
        elif args.command == "train":
            train_command(args)
            
        elif args.command == "evolve":
            # 進化的学習
            agent.run_evolution(
                currency_pair=args.pair,
                model_type=args.model,
                generations=args.generations,
                population_size=args.population,
                parallel=args.parallel
            )
            
        elif args.command == "backtest":
            # バックテスト
            agent.run_backtest(args.model_id)
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
    finally:
        agent.shutdown()


def train_command(args):
    """モデル学習コマンドの実行
    
    Args:
        args: コマンドライン引数
    """
    try:
        # FX学習エージェントの初期化
        agent = FXLearningAgent()
        agent.initialize()
        
        # データの取得（1年分）
        logger.info(f"{args.pair}の1年分のデータを取得しています...")
        train_data = agent.fetch_data(args.pair, "1h", 365)  # 1年分のデータ
        
        if train_data is None or len(train_data) == 0:
            raise ValueError("学習データが取得できませんでした")
        
        # データを学習用と検証用に分割（80:20）
        split_idx = int(len(train_data) * 0.8)
        train_data, val_data = train_data[:split_idx], train_data[split_idx:]
        
        logger.info(f"学習データ: {len(train_data)}件, 検証データ: {len(val_data)}件")
        
        # モデルタイプの処理
        model_type = args.model.lower()
        if model_type not in ["lstm", "sentiment"]:
            raise ValueError(f"サポートされていないモデルタイプ: {args.model}")
        
        # モデルの学習
        logger.info(f"{model_type}モデルの学習を開始します...")
        result = agent.train(
            currency_pair=args.pair,
            model_type=model_type
        )
        
        if result:
            if "error" in result:
                logger.error(f"モデル学習に失敗しました: {result['error']}")
            else:
                logger.info(f"モデル学習が完了しました。モデルID: {result.get('model_saved')}")
                logger.info(f"学習結果: {result}")
                
                # モデルの評価
                logger.info("モデルの評価を実行しています...")
                agent.evaluate(result.get('model_saved'))
        else:
            logger.error("モデル学習に失敗しました: 結果が返されませんでした")
            
    except Exception as e:
        logger.error(f"コマンド実行エラー: {str(e)}", exc_info=True)
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main() 