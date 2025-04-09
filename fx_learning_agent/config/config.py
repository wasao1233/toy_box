import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# .envファイルがあれば読み込む
load_dotenv()

@dataclass
class CurrencyPair:
    """通貨ペア設定クラス"""
    symbol: str  # 例: "USD/JPY"
    display_name: str
    pip_value: float
    enabled: bool = True


@dataclass
class DataSourceConfig:
    """データソース設定クラス"""
    name: str
    enabled: bool
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseConfig:
    """データベース設定クラス"""
    engine: str = "postgresql"  # "postgresql", "sqlite", etc.
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "fx_learning"
    schema: Optional[str] = None


@dataclass
class LearningConfig:
    """学習設定クラス"""
    initial_population: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    fitness_metrics: List[str] = field(default_factory=lambda: ["profit", "win_rate", "sharpe_ratio"])
    backtest_days: int = 30
    validation_days: int = 7


@dataclass
class Config:
    """メイン設定クラス"""
    # 通貨ペア設定
    currency_pairs: List[CurrencyPair] = field(default_factory=list)
    
    # データソース設定
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    
    # データベース設定
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # 学習設定
    learning: LearningConfig = field(default_factory=LearningConfig)
    
    # ログ設定
    log_level: str = "INFO"
    log_file: str = "fx_learning.log"
    
    # その他設定
    save_models_path: str = "models/saved"
    threads: int = max(1, os.cpu_count() - 1) if os.cpu_count() else 2
    random_seed: int = 42


def get_default_config() -> Config:
    """デフォルト設定を生成"""
    config = Config()
    
    # 通貨ペア設定
    config.currency_pairs = [
        CurrencyPair(symbol="USD/JPY", display_name="米ドル/日本円", pip_value=0.01),
        CurrencyPair(symbol="EUR/USD", display_name="ユーロ/米ドル", pip_value=0.0001),
        CurrencyPair(symbol="GBP/USD", display_name="英ポンド/米ドル", pip_value=0.0001),
        CurrencyPair(symbol="AUD/USD", display_name="豪ドル/米ドル", pip_value=0.0001),
        CurrencyPair(symbol="USD/CHF", display_name="米ドル/スイスフラン", pip_value=0.0001),
        CurrencyPair(symbol="EUR/JPY", display_name="ユーロ/日本円", pip_value=0.01),
        CurrencyPair(symbol="GBP/JPY", display_name="英ポンド/日本円", pip_value=0.01)
    ]
    
    # AlphaVantage設定
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "your_api_key_here")
    config.data_sources["alpha_vantage"] = DataSourceConfig(
        name="Alpha Vantage",
        enabled=True,
        api_key=alpha_vantage_key,
        base_url="https://www.alphavantage.co/query",
        rate_limit=5,  # 1分あたり最大5リクエスト（無料プラン）
        params={
            "forex_daily_function": "FX_DAILY",
            "forex_intraday_function": "FX_INTRADAY",
            "output_size": "full"
        }
    )
    
    # NewsAPI設定
    news_api_key = os.getenv("NEWS_API_KEY", "your_api_key_here")
    config.data_sources["news_api"] = DataSourceConfig(
        name="News API",
        enabled=True,
        api_key=news_api_key,
        base_url="https://newsapi.org/v2",
        rate_limit=100,  # 1日あたり100リクエスト（無料プラン）
        params={
            "endpoints": {
                "top_headlines": "/top-headlines",
                "everything": "/everything"
            },
            "default_language": "en",
            "default_page_size": 100
        }
    )
    
    # データベース設定
    db_engine = os.getenv("DB_ENGINE", "postgresql")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_username = os.getenv("DB_USERNAME", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_name = os.getenv("DB_NAME", "fx_learning")
    db_schema = os.getenv("DB_SCHEMA", "public")
    
    config.database = DatabaseConfig(
        engine=db_engine,
        host=db_host,
        port=db_port,
        username=db_username,
        password=db_password,
        database=db_name,
        schema=db_schema
    )
    
    return config


_config = None

def get_config() -> Config:
    """設定を取得"""
    global _config
    if _config is None:
        _config = get_default_config()
    return _config


def load_config_from_file(file_path: str) -> Config:
    """ファイルから設定を読み込む"""
    global _config
    try:
        # TODO: ファイルから設定を読み込む実装を追加
        # ここではデフォルト設定を返す
        _config = get_default_config()
        return _config
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {str(e)}")
        # エラーが発生した場合はデフォルト設定を使用
        _config = get_default_config()
        return _config 