import os
import logging
import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

from config.config import get_config

def get_logger(name: str, log_level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """指定された名前のロガーを取得または作成します。

    Args:
        name: ロガー名
        log_level: ログレベル。指定されなければ設定ファイルの値を使用
        log_file: ログファイルパス。指定されなければ設定ファイルの値を使用

    Returns:
        設定済みのロガーインスタンス
    """
    config = get_config()
    
    # ログレベルとファイルの決定
    if log_level is None:
        log_level = config.log_level
    
    if log_file is None:
        log_file = config.log_file
    
    # ログディレクトリの作成
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # ロガーの取得
    logger = logging.getLogger(name)
    
    # 既に設定済みの場合は再利用
    if logger.handlers:
        return logger
    
    # ログレベルの設定
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # フォーマッタの作成
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ファイルハンドラの設定
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger 