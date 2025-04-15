from utils.database import get_db
from utils.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)

def test_db_connection():
    """データベース接続のテスト"""
    try:
        # データベース接続の取得
        db = get_db()
        session = db.get_session()
        
        # ニュース記事の取得
        news_count = session.execute(text("SELECT COUNT(*) FROM news_items")).scalar()
        logger.info(f"ニュース記事数: {news_count}件")
        
        # 為替レートの取得
        rates_count = session.execute(text("SELECT COUNT(*) FROM currency_rates")).scalar()
        logger.info(f"為替レート数: {rates_count}件")
        
        # 最新のニュース記事
        latest_news = session.execute(text("""
            SELECT title, published_at, source
            FROM news_items
            ORDER BY published_at DESC
            LIMIT 3
        """)).fetchall()
        
        logger.info("最新のニュース記事:")
        for news in latest_news:
            logger.info(f"- {news.title} ({news.source}, {news.published_at})")
        
        # 最新の為替レート
        latest_rates = session.execute(text("""
            SELECT timestamp, symbol, close
            FROM currency_rates
            ORDER BY timestamp DESC
            LIMIT 3
        """)).fetchall()
        
        logger.info("最新の為替レート:")
        for rate in latest_rates:
            logger.info(f"- {rate.symbol}: {rate.close} ({rate.timestamp})")
        
    except Exception as e:
        logger.error(f"データベース接続エラー: {str(e)}", exc_info=True)
    finally:
        session.close()

if __name__ == "__main__":
    test_db_connection() 