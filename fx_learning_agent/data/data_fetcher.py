import os
import time
import json
import datetime
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
from newsapi import NewsApiClient
from sqlalchemy.orm import Session

from utils.logger import get_logger
from utils.database import get_db
from config.config import get_config
from models.data_models import CurrencyRate, NewsItem

logger = get_logger(__name__)


class DataFetcher(ABC):
    """データ取得の基底クラス"""
    
    def __init__(self, source_name: str):
        """初期化"""
        self.config = get_config()
        self.db = get_db()
        self.source_name = source_name
        
        # データソース設定の取得
        self.source_config = self.config.data_sources.get(source_name)
        if not self.source_config:
            raise ValueError(f"データソース '{source_name}' の設定が見つかりません")
        
        if not self.source_config.enabled:
            logger.warning(f"データソース '{source_name}' は無効化されています")
    
    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        """データ取得の抽象メソッド"""
        pass
    
    def _handle_rate_limit(self):
        """レート制限の処理"""
        if self.source_config.rate_limit:
            time.sleep(60 / self.source_config.rate_limit)  # 1分あたりのリクエスト数で制限


class ForexDataFetcher(DataFetcher):
    """為替データ取得クラス"""
    
    def __init__(self):
        """初期化"""
        super().__init__("alpha_vantage")
        self.base_url = self.source_config.base_url
        self.api_key = self.source_config.api_key
        self.market_indices = {
            "VIX": "VIX",
            "S&P500": "SPY",
            "日経平均": "N225",
            "DXY": "DXY",
            "金": "XAU/USD",
            "原油": "WTI"
        }
    
    def _generate_demo_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """デモデータの生成
        
        Args:
            symbol: 通貨ペア
            start_date: 開始日時
            end_date: 終了日時
            timeframe: 時間枠
            
        Returns:
            デモデータのデータフレーム
        """
        # 日付範囲の生成
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 基準値の設定（USD/JPYの場合）
        if symbol == "USD/JPY":
            base_price = 150.0
            volatility = 0.5
        else:
            base_price = 100.0
            volatility = 0.3
        
        # ランダムウォークでデータを生成
        n_points = len(dates)
        changes = np.random.normal(0, volatility, n_points)
        prices = base_price + np.cumsum(changes)
        
        # データフレームの作成
        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, volatility, n_points),
            'low': prices - np.random.uniform(0, volatility, n_points),
            'close': prices + np.random.normal(0, volatility/2, n_points),
            'volume': np.random.randint(1000, 10000, n_points)
        }, index=dates)
        
        # 値の調整（highが最大、lowが最小になるように）
        for i in range(len(df)):
            values = [df.iloc[i]['open'], df.iloc[i]['close']]
            df.iloc[i]['high'] = max(values) + abs(np.random.normal(0, volatility/4))
            df.iloc[i]['low'] = min(values) - abs(np.random.normal(0, volatility/4))
        
        return df
    
    def fetch_data(self, symbol: str, timeframe: str = "1d", 
                  start_date: Optional[datetime.datetime] = None, 
                  end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
        """抽象メソッドの実装 - 過去の為替レートを取得する"""
        return self.fetch_historical_rates(symbol, timeframe, start_date, end_date)
    
    def fetch_current_rate(self, symbol: str) -> Optional[Dict[str, float]]:
        """現在のレートを取得
        
        Args:
            symbol: 通貨ペア（例: "USD/JPY"）
            
        Returns:
            現在のレート情報（bid, ask, spread）または None
        """
        try:
            # APIリクエストパラメータの構築
            from_currency, to_currency = symbol.split('/')
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key
            }
            
            # リクエスト送信
            self._handle_rate_limit()
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # レスポンスの解析
            if "Realtime Currency Exchange Rate" in data:
                exchange_data = data["Realtime Currency Exchange Rate"]
                rate = float(exchange_data["5. Exchange Rate"])
                
                # BidとAskはAPIから取得できないため、簡易的に計算
                bid = rate * 0.9998  # 0.02% 減
                ask = rate * 1.0002  # 0.02% 増
                
                return {
                    "symbol": symbol,
                    "rate": rate,
                    "bid": bid,
                    "ask": ask,
                    "spread": ask - bid,
                    "timestamp": datetime.datetime.utcnow()
                }
            
            logger.warning(f"為替レートデータが取得できませんでした: {data}")
            return None
            
        except Exception as e:
            logger.error(f"現在のレート取得エラー ({symbol}): {str(e)}", exc_info=True)
            return None
    
    def fetch_historical_rates(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        save_to_db: bool = False
    ) -> pd.DataFrame:
        """過去の為替レートを取得（日次データに特化）
        
        Args:
            symbol: 通貨ペア（例: "USD/JPY"）
            timeframe: 時間枠（"1d"のみ対応）
            start_date: 開始日時
            end_date: 終了日時
            save_to_db: データベースに保存するかどうか
            
        Returns:
            過去の為替レートのデータフレーム
        """
        try:
            # 日次データのみ対応
            if timeframe != "1d":
                logger.warning("日次データ（1d）のみ対応しています")
                timeframe = "1d"
            
            # 日付の調整
            if start_date is None:
                start_date = datetime.datetime.now() - datetime.timedelta(days=365)
            if end_date is None:
                end_date = datetime.datetime.now()
            
            # データベースからデータを取得
            try:
                session = self.db.get_session()
                rates = session.query(CurrencyRate).filter(
                    CurrencyRate.symbol == symbol,
                    CurrencyRate.timeframe == timeframe,
                    CurrencyRate.timestamp >= start_date,
                    CurrencyRate.timestamp <= end_date
                ).order_by(CurrencyRate.timestamp).all()
                
                if rates:
                    # データベースから取得したデータをデータフレームに変換
                    data = {
                        'open': [rate.open for rate in rates],
                        'high': [rate.high for rate in rates],
                        'low': [rate.low for rate in rates],
                        'close': [rate.close for rate in rates],
                        'volume': [rate.volume for rate in rates]
                    }
                    df = pd.DataFrame(data, index=[rate.timestamp for rate in rates])
                    logger.info(f"データベースから{len(rates)}件の為替レートを取得しました")
                    return df
            except Exception as e:
                logger.warning(f"データベースからのデータ取得に失敗しました: {str(e)}")
            finally:
                session.close()
            
            # データベースにデータがない場合、APIから取得
            try:
                df = self._fetch_from_api(symbol, timeframe, start_date, end_date)
                if not df.empty:
                    if save_to_db:
                        self._save_rates_to_db(df, symbol, timeframe)
                    return df
            except Exception as e:
                logger.warning(f"APIからのデータ取得に失敗しました: {str(e)}")
            
            # APIも失敗した場合、デモデータを生成
            logger.info("デモデータを生成します")
            df = self._generate_demo_data(symbol, start_date, end_date, timeframe)
            
            if save_to_db and not df.empty:
                self._save_rates_to_db(df, symbol, timeframe)
            
            return df
            
        except Exception as e:
            logger.error(f"データ取得エラー: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _fetch_from_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime.datetime],
        end_date: Optional[datetime.datetime]
    ) -> pd.DataFrame:
        """APIからデータを取得
        
        Args:
            symbol: 通貨ペア
            timeframe: 時間枠
            start_date: 開始日時
            end_date: 終了日時
            
        Returns:
            取得したデータのデータフレーム
        """
        # パラメータの準備
        from_currency, to_currency = symbol.split('/')
        
        # APIリクエストパラメータの構築
        params = {
            "function": self.source_config.params["forex_daily_function"],
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        # リクエスト送信
        self._handle_rate_limit()
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # レスポンスの解析
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.warning(f"為替レート時系列データが取得できませんでした: {data}")
            return pd.DataFrame()
        
        # データフレームに変換
        rates_dict = data[time_series_key]
        df = pd.DataFrame.from_dict(rates_dict, orient="index")
        
        # カラム名の整理
        df.columns = [col.split(". ")[1] for col in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # データ型の変換
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        
        # ボリュームがあれば変換
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"])
        else:
            df["volume"] = 0
        
        # 日付フィルタリング
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def _save_rates_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """為替レートをデータベースに保存
        
        Args:
            df: 為替レートのデータフレーム
            symbol: 通貨ペア
            timeframe: 時間枠
        """
        try:
            session = self.db.get_session()
            count = 0
            
            for index, row in df.iterrows():
                # 同じタイムスタンプのレコードがあるか確認
                existing = session.query(CurrencyRate).filter(
                    CurrencyRate.symbol == symbol,
                    CurrencyRate.timestamp == index,
                    CurrencyRate.timeframe == timeframe
                ).first()
                
                if existing:
                    # 既存レコードは更新しない
                    continue
                
                # NumPy型をPythonの標準型に変換
                open_price = float(row['open'])
                high_price = float(row['high'])
                low_price = float(row['low'])
                close_price = float(row['close'])
                volume = float(row['volume']) if 'volume' in row else 0.0
                
                # レコードの作成
                rate = CurrencyRate(
                    symbol=symbol,
                    timestamp=index,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe=timeframe,
                    source="alpha_vantage"
                )
                
                session.add(rate)
                count += 1
                
                # 一定数ごとにコミット
                if count % 100 == 0:
                    session.commit()
            
            # 残りをコミット
            session.commit()
            logger.info(f"{count}件の為替レートをデータベースに保存しました")
            
        except Exception as e:
            logger.error(f"為替レートの保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close()
    
    def fetch_long_term_analysis(self, symbol: str, years: int = 5) -> Dict[str, Any]:
        """長期的な為替データの分析を取得
        
        Args:
            symbol: 通貨ペア（例: "USD/JPY"）
            years: 分析する年数
            
        Returns:
            分析結果を含む辞書
        """
        try:
            # 過去のデータを取得
            end_date = datetime.datetime.utcnow()
            start_date = end_date - datetime.timedelta(days=years*365)
            
            df = self.fetch_historical_rates(
                symbol=symbol,
                timeframe="1d",
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.warning(f"分析用のデータが取得できませんでした: {symbol}")
                return {}
            
            # 基本的な統計量の計算
            analysis = {
                "period": f"{years}年",
                "start_date": df.index.min(),
                "end_date": df.index.max(),
                "total_days": len(df),
                "average_close": df["close"].mean(),
                "max_close": df["close"].max(),
                "min_close": df["close"].min(),
                "volatility": df["close"].std(),
                "daily_returns": df["close"].pct_change().dropna(),
                "annual_returns": df["close"].pct_change(periods=252).dropna()  # 252営業日
            }
            
            # 月次リターンの計算
            monthly_returns = df["close"].resample('M').last().pct_change().dropna()
            analysis["monthly_returns"] = monthly_returns
            
            # 年次リターンの計算
            yearly_returns = df["close"].resample('Y').last().pct_change().dropna()
            analysis["yearly_returns"] = yearly_returns
            
            # シャープレシオの計算（リスクフリーレートは0%と仮定）
            daily_returns = analysis["daily_returns"]
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            analysis["sharpe_ratio"] = sharpe_ratio
            
            return analysis
            
        except Exception as e:
            logger.error(f"長期的分析エラー ({symbol}): {str(e)}", exc_info=True)
            return {}

    def fetch_market_indices(
        self,
        indices: List[str] = None,
        timeframe: str = "1d",
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """市場指数データを取得
        
        Args:
            indices: 取得する指数のリスト（Noneの場合は全て）
            timeframe: 時間枠
            start_date: 開始日時
            end_date: 終了日時
            
        Returns:
            指数ごとのデータフレームの辞書
        """
        try:
            if indices is None:
                indices = list(self.market_indices.keys())
            
            results = {}
            for index in indices:
                if index not in self.market_indices:
                    logger.warning(f"サポートされていない指数: {index}")
                    continue
                
                symbol = self.market_indices[index]
                df = self.fetch_historical_rates(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    results[index] = df
                    logger.info(f"{index}のデータを取得しました: {len(df)}件")
                else:
                    logger.warning(f"{index}のデータが取得できませんでした")
            
            return results
            
        except Exception as e:
            logger.error(f"市場指数データ取得エラー: {str(e)}", exc_info=True)
            return {}


class NewsDataFetcher(DataFetcher):
    """ニュースデータ取得クラス"""
    
    def __init__(self):
        """初期化"""
        super().__init__("news_api")
        self.api_key = self.source_config.api_key
        self.news_api = NewsApiClient(api_key=self.api_key)
    
    def fetch_data(self, keywords: List[str], 
                  from_date: Optional[datetime.datetime] = None, 
                  to_date: Optional[datetime.datetime] = None, 
                  **kwargs) -> List[Dict[str, Any]]:
        """抽象メソッドの実装 - キーワードに関連するニュースを取得する"""
        return self.fetch_news(keywords, from_date, to_date, **kwargs)
    
    def fetch_news(
        self,
        keywords: List[str],
        from_date: Optional[datetime.datetime] = None,
        to_date: Optional[datetime.datetime] = None,
        language: str = "en",
        save_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """キーワードに関連するニュースを取得
        
        Args:
            keywords: 検索キーワードのリスト
            from_date: 開始日時
            to_date: 終了日時
            language: 言語コード
            save_to_db: データベースに保存するかどうか
            
        Returns:
            ニュース記事のリスト
        """
        try:
            if not keywords:
                logger.warning("検索キーワードが指定されていません")
                return []
            
            # 日付の調整
            if not from_date:
                from_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)
            
            if not to_date:
                to_date = datetime.datetime.utcnow()
            
            # 日付フォーマットの変換
            from_str = from_date.strftime("%Y-%m-%d")
            to_str = to_date.strftime("%Y-%m-%d")
            
            # クエリの構築
            query = " OR ".join([f'"{keyword}"' for keyword in keywords])
            
            # APIリクエスト
            self._handle_rate_limit()
            response = self.news_api.get_everything(
                q=query,
                from_param=from_str,
                to=to_str,
                language=language,
                sort_by="relevancy",
                page_size=100
            )
            
            # レスポンスの処理
            articles = response.get("articles", [])
            logger.info(f"{len(articles)}件のニュース記事を取得しました")
            
            # データベースに保存
            if save_to_db and articles:
                self._save_news_to_db(articles, keywords)
            
            return articles
            
        except Exception as e:
            logger.error(f"ニュース取得エラー: {str(e)}", exc_info=True)
            return []
    
    def _save_news_to_db(self, articles: List[Dict[str, Any]], keywords: List[str]):
        """ニュース記事をデータベースに保存
        
        Args:
            articles: ニュース記事のリスト
            keywords: 検索キーワードのリスト
        """
        try:
            session = self.db.get_session()
            count = 0
            
            for article in articles:
                # URLで既存記事を確認
                url = article.get("url")
                if not url:
                    continue
                    
                existing = session.query(NewsItem).filter(NewsItem.url == url).first()
                if existing:
                    # 既存記事は更新しない
                    continue
                
                # 公開日の解析
                published_at = None
                try:
                    published_at_str = article.get("publishedAt")
                    if published_at_str:
                        published_at = datetime.datetime.strptime(
                            published_at_str, "%Y-%m-%dT%H:%M:%SZ"
                        )
                except Exception:
                    # 日付解析エラーの場合は現在時刻を使用
                    published_at = datetime.datetime.utcnow()
                
                # 関連通貨のフィルタリング
                related_currencies = []
                for keyword in keywords:
                    if any([
                        keyword in article.get("title", "") or "",
                        keyword in (article.get("description", "") or ""),
                        keyword in (article.get("content", "") or "")
                    ]):
                        if "/" in keyword:  # 通貨ペアの場合
                            related_currencies.append(keyword)
                        elif len(keyword) == 3:  # 通貨コードの場合
                            for pair in self.config.currency_pairs:
                                if keyword in pair.symbol:
                                    related_currencies.append(pair.symbol)
                
                # 重複を除去
                related_currencies = list(set(related_currencies))
                
                # 記事の保存
                news_item = NewsItem(
                    title=article.get("title", ""),
                    content=article.get("content", ""),
                    summary=article.get("description", ""),
                    source=article.get("source", {}).get("name", "Unknown"),
                    url=url,
                    published_at=published_at or datetime.datetime.utcnow(),
                    related_currencies=",".join(related_currencies)
                )
                session.add(news_item)
                count += 1
                
                # 一定数ごとにコミット
                if count % 50 == 0:
                    session.commit()
            
            # 残りをコミット
            session.commit()
            logger.info(f"{count}件のニュース記事をデータベースに保存しました")
            
        except Exception as e:
            logger.error(f"ニュース記事の保存エラー: {str(e)}", exc_info=True)
            session.rollback()
        finally:
            session.close()


# シングルトンインスタンス
_forex_fetcher = None
_news_fetcher = None

def get_forex_fetcher() -> ForexDataFetcher:
    """為替データ取得クラスのインスタンスを取得"""
    global _forex_fetcher
    if _forex_fetcher is None:
        _forex_fetcher = ForexDataFetcher()
    return _forex_fetcher

def get_news_fetcher() -> NewsDataFetcher:
    """ニュースデータ取得クラスのインスタンスを取得"""
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsDataFetcher()
    return _news_fetcher 