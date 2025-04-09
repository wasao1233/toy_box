import os
from typing import Optional
import logging
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from config.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# SQLAlchemyのベースクラス
Base = declarative_base()

class Database:
    """データベース接続管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.config = get_config().database
        self.engine = None
        self.session_factory = None
        self.session = None
        self.initialized = False
    
    def initialize(self):
        """データベース接続の初期化"""
        if self.initialized:
            return
        
        try:
            # 接続文字列の作成
            if self.config.engine == 'postgresql':
                connection_string = (
                    f"postgresql://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port}/{self.config.database}"
                )
            elif self.config.engine == 'sqlite':
                # SQLiteの場合は相対パスで指定
                db_dir = os.path.join(os.getcwd(), 'data')
                os.makedirs(db_dir, exist_ok=True)
                connection_string = f"sqlite:///{os.path.join(db_dir, self.config.database)}.db"
            else:
                raise ValueError(f"サポートされていないデータベースエンジン: {self.config.engine}")
            
            # エンジンの作成
            self.engine = create_engine(
                connection_string,
                echo=False,  # SQLログを出力するかどうか
                pool_pre_ping=True,  # 接続前にpingを実行して確認
                pool_recycle=3600  # 接続を1時間ごとにリサイクル
            )
            
            # セッションファクトリの作成
            self.session_factory = sessionmaker(bind=self.engine)
            self.session = scoped_session(self.session_factory)
            
            logger.info(f"{self.config.engine}データベースに接続しました: {self.config.database}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"データベース接続エラー: {str(e)}", exc_info=True)
            raise
    
    def create_tables(self):
        """テーブルの作成"""
        if not self.initialized:
            self.initialize()
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("データベーステーブルを作成しました")
        except Exception as e:
            logger.error(f"テーブル作成エラー: {str(e)}", exc_info=True)
            raise
    
    def get_session(self):
        """セッションの取得"""
        if not self.initialized:
            self.initialize()
        return self.session
    
    def dispose(self):
        """接続のクローズ"""
        if self.session:
            self.session.remove()
        
        if self.engine:
            self.engine.dispose()
            logger.info("データベース接続をクローズしました")
        
        self.initialized = False


# シングルトンインスタンス
_db_instance = None

def get_db() -> Database:
    """データベースインスタンスの取得"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance 