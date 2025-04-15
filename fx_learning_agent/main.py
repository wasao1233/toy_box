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
    
    def __init__(self):
        """初期化"""
        self.config = get_config()
        self.db = get_db()
        self.forex_fetcher = get_forex_fetcher()
        self.news_fetcher = get_news_fetcher()
        self.genetic_algorithm = GeneticAlgorithm()
        
        # センチメントモデル
        self.sentiment_model = SentimentModel()
        
        # シャットダウンフラグ
        self.shutdown_requested = False
        
        # 並列処理設定
        self.n_workers = self.config.threads
        logger.info(f"FX学習エージェントを初期化しました (ワーカー数: {self.n_workers})")
    
    def initialize(self):
        """システムの初期化"""
        # データベースの初期化
        self.db.initialize()
        self.db.create_tables()
        
        # センチメントモデルの初期化
        try:
            self.sentiment_model.initialize()
        except Exception as e:
            logger.error(f"センチメントモデルの初期化エラー: {str(e)}", exc_info=True)
        
        logger.info("FX学習エージェントの初期化が完了しました")
    
    def fetch_data(self, currency_pair: str, timeframe: str, days: int = 30):
        """データ取得
        
        Args:
            currency_pair: 通貨ペア
            timeframe: 時間枠
            days: 取得日数
        """
        try:
            logger.info(f"為替データの取得: {currency_pair}, {timeframe}, {days}日間")
            
            # 日付範囲の設定
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # 為替データの取得
            df = self.forex_fetcher.fetch_historical_rates(
                symbol=currency_pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                save_to_db=True
            )
            
            if df.empty:
                logger.warning(f"為替データが取得できませんでした: {currency_pair}, {timeframe}")
                return None
            
            logger.info(f"{len(df)}件の為替データを取得しました: {currency_pair}, {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"データ取得エラー: {str(e)}", exc_info=True)
            return None
    
    def fetch_news(self, keywords: List[str], days: int = 7):
        """ニュースデータ取得
        
        Args:
            keywords: 検索キーワード
            days: 取得日数
        """
        try:
            logger.info(f"ニュースデータの取得: {keywords}, {days}日間")
            
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
                logger.warning(f"ニュースデータが取得できませんでした: {keywords}")
                return None
            
            logger.info(f"{len(articles)}件のニュースデータを取得しました")
            
            # 市場指数データの取得
            market_indices = self.forex_fetcher.fetch_market_indices(
                timeframe="1d",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_indices:
                logger.info(f"{len(market_indices)}件の市場指数データを取得しました")
            
            # センチメント分析
            if hasattr(self, 'sentiment_model') and self.sentiment_model.model is not None:
                self.sentiment_model.analyze_news_sentiment(articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"ニュース取得エラー: {str(e)}", exc_info=True)
            return None
    
    def train_model(
        self,
        model_type: str,
        train_data: Any,
        val_data: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """モデルの学習
        
        Args:
            model_type: モデルタイプ ("lstm" または "sentiment")
            train_data: 学習データ
            val_data: 検証データ
            model_config: モデルの設定
            
        Returns:
            学習結果と評価情報
        """
        try:
            model_type = model_type.lower()  # 小文字に統一
            
            # データの検証
            if train_data is None or len(train_data) == 0:
                raise ValueError("学習データが空です")
            
            if val_data is not None and len(val_data) == 0:
                raise ValueError("検証データが空です")
            
            # モデル設定の取得
            if model_config is None:
                if model_type == "lstm":
                    model_config = get_lstm_config()
                elif model_type == "sentiment":
                    model_config = get_sentiment_config()
                else:
                    raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
            
            # モデルの初期化
            if model_type == "lstm":
                model = TimeSeriesModel(config=model_config.to_dict())
            elif model_type == "sentiment":
                model = SentimentModel(config=model_config.to_dict())
                model.initialize()
                model._save_model_metadata()
                return {
                    "status": "success",
                    "model_type": model_type,
                    "model_id": model.db_model_id,
                    "model_uuid": model_config.uuid,
                    "results": {"note": "事前学習済みモデルを使用"}
                }
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
            
            # 学習の実行
            train_results = model.train(
                df_train=train_data,
                df_val=val_data,
                save_model=True
            )
            
            # 評価
            if val_data is not None:
                evaluation = model.evaluate(val_data, save_to_db=True)
                train_results.update(evaluation)
            
            return {
                "status": "success",
                "model_type": model_type,
                "model_id": model.db_model_id,
                "model_uuid": model_config.uuid,
                "results": train_results
            }
            
        except Exception as e:
            logger.error(f"モデル学習エラー: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
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
            
            logger.info(f"進化的学習の開始: {currency_pair}, {model_type}, "
                       f"世代数={generations}, 集団サイズ={population_size}")
            
            # データの取得
            train_data = self.fetch_data(currency_pair, "1h", days=60)
            val_data = self.fetch_data(currency_pair, "1h", days=10)
            
            if train_data is None or val_data is None:
                logger.error("学習データが取得できませんでした")
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
                logger.error(f"サポートされていないモデルタイプ: {model_type}")
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
                logger.info(f"世代 {generation+1}/{generations} の学習開始")
                
                # 各モデルの学習と評価（並列処理）
                train_results = []
                
                if parallel and self.n_workers > 1:
                    # 並列処理
                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        futures = []
                        for model_config in population:
                            future = executor.submit(
                                self.train_model,
                                model_config,
                                train_data,
                                val_data
                            )
                            futures.append(future)
                        
                        # 結果の収集
                        for future in futures:
                            train_results.append(future.result())
                else:
                    # 順次処理
                    for model_config in population:
                        result = self.train_model(model_config, train_data, val_data)
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
                
                logger.info(f"世代 {generation+1}/{generations} の学習完了")
            
            logger.info(f"進化的学習完了: {generations}世代, {len(population)}モデル")
            
        except Exception as e:
            logger.error(f"進化的学習エラー: {str(e)}", exc_info=True)
    
    def run_backtest(self, model_id: int, test_data: Any = None):
        """バックテストの実行
        
        Args:
            model_id: モデルID
            test_data: テストデータ（指定がなければ新たに取得）
        """
        # TODO: バックテストの実装
        logger.warning("バックテスト機能は未実装です")
    
    def shutdown(self):
        """エージェントのシャットダウン"""
        logger.info("FX学習エージェントをシャットダウンしています...")
        self.shutdown_requested = True
        
        # データベース接続のクローズ
        if hasattr(self, 'db') and self.db:
            self.db.dispose()
        
        logger.info("FX学習エージェントをシャットダウンしました")

    def evaluate_model(self, model_id: int, test_data: Any) -> Dict[str, Any]:
        """モデルの評価を実行
        
        Args:
            model_id: モデルID
            test_data: テストデータ
            
        Returns:
            評価結果
        """
        try:
            # モデルの読み込み
            model = self._load_model(model_id)
            if model is None:
                raise ValueError(f"モデルが見つかりません: ID={model_id}")
            
            # 評価の実行
            evaluation = model.evaluate(test_data, save_to_db=True)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"モデル評価エラー: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _load_model(self, model_id: int) -> Optional[Any]:
        """モデルをデータベースから読み込む
        
        Args:
            model_id: モデルID
            
        Returns:
            読み込んだモデル
        """
        try:
            session = self.db.get_session()
            model_record = session.query(DbModel).filter_by(id=model_id).first()
            
            if model_record is None:
                return None
            
            # モデルタイプに応じて適切なモデルクラスをインスタンス化
            if model_record.model_type.lower() == "lstm":
                model = TimeSeriesModel()
            elif model_record.model_type.lower() == "sentiment":
                model = SentimentModel()
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_record.model_type}")
            
            # モデルの読み込み
            model.load_model(model_id)
            
            return model
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {str(e)}", exc_info=True)
            return None
        finally:
            session.close()


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
        result = agent.train_model(
            model_type=model_type,
            train_data=train_data,
            val_data=val_data
        )
        
        if result["status"] == "success":
            logger.info(f"モデル学習が完了しました。モデルID: {result['model_id']}")
            logger.info(f"学習結果: {result['results']}")
            
            # モデルの評価
            logger.info("モデルの評価を実行しています...")
            evaluation = agent.evaluate_model(
                model_id=result['model_id'],
                test_data=val_data
            )
            logger.info(f"評価結果: {evaluation}")
            
        else:
            logger.error(f"モデル学習に失敗しました: {result['error']}")
            
    except Exception as e:
        logger.error(f"コマンド実行エラー: {str(e)}", exc_info=True)
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main() 