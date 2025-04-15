from utils.database import Base, get_db
from models.data_models import *  # 全てのモデルクラスをインポート

def reset_database():
    """データベースのテーブルを再作成する"""
    db = get_db()
    db.initialize()
    
    # 既存のテーブルを削除
    Base.metadata.drop_all(db.engine)
    print("既存のテーブルを削除しました")
    
    # テーブルを再作成
    Base.metadata.create_all(db.engine)
    print("テーブルを再作成しました")
    
    # 接続をクローズ
    db.dispose()
    print("データベース接続をクローズしました")

if __name__ == "__main__":
    reset_database() 